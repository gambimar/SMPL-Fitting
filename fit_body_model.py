
import torch
import numpy as np
from tqdm import tqdm
import os
import argparse


from losses import ChamferDistance, MaxMixturePrior, summed_L2, LossTracker, SMPL_to_SKEL_vert_distance
from visualization import viz_error_curves, viz_iteration, set_init_plot, viz_final_fit
from utils import (check_scan_prequisites_fit_bm, cleanup, 
                   load_config, save_configs,
                   load_landmarks,load_scan,
                   get_already_fitted_scan_names, get_skipped_scan_names, 
                   initialize_fit_bm_loss_weights, load_loss_weights_config, 
                   print_loss_weights, print_losses, print_params, 
                   process_body_model_path, process_default_dtype, 
                   process_landmarks, process_visualize_steps, 
                   create_results_directory, to_txt,
                   setup_socket, send_to_socket,
                   )
from body_models import BodyModel
from body_parameters import BodyParameters
from datasets import FAUST, CAESAR

#from dash_app import run_dash_app_as_subprocess
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_body_model(input_dict: dict, cfg: dict):
    """
    Fit a body body model (SMPL/SMPLX) to the input scan using
    data, landmark and prior losses

    :param: input_dict (dict): with keys:
            "name": name of the scan
            "vertices": numpy array (N,3)
            "faces": numpy array (N,3) or None if no faces
            "landmarks": dictionary with keys as landmark names and 
                        values as list [x,y,z] or np.ndarray (3,)
    :param: cfg (dict): config file defined in configs/config.yaml
    """
    DEFAULT_DTYPE = cfg['default_dtype']
    VERBOSE = cfg['verbose']
    VISUALIZE = cfg['visualize']
    VISUALIZE_STEPS = cfg['visualize_steps']
    VISUALIZE_LOGSCALE = cfg["error_curves_logscale"]
    SAVE_PATH = cfg['save_path']
    SOCKET_TYPE = cfg["socket_type"]
    VOLUME_TARGET = cfg["volume_target"]

    if VISUALIZE:
        socket = cfg["socket"]

    # inputs
    input_name = input_dict["name"]
    input_vertices = input_dict["vertices"]
    input_faces = input_dict["faces"]
    input_landmarks = input_dict["landmarks"]
    input_index = input_dict["scan_index"]

    # process inputs
    input_vertices = torch.from_numpy(input_vertices).type(DEFAULT_DTYPE).unsqueeze(0).to(DEVICE)
    input_faces = torch.from_numpy(input_faces).type(DEFAULT_DTYPE) if \
                            (not isinstance(input_faces,type(None))) else None

    landmarks_order = sorted(list(input_landmarks.keys()))
    input_landmarks = np.array([input_landmarks[k] for k in landmarks_order])
    input_landmarks = torch.from_numpy(input_landmarks)
    input_landmarks = input_landmarks.type(DEFAULT_DTYPE).to(DEVICE)

    # setup body model
    body_model = BodyModel(cfg)
    
    if torch.cuda.is_available():
        body_model.cuda()
    body_model_params = BodyParameters(cfg).to(DEVICE)
    body_model_landmark_inds = body_model.landmark_indices(landmarks_order)
    print(f"Using {len(input_landmarks)}/{len(body_model.all_landmark_indices)} landmarks.")

    print('Fitting body model to scan:', input_name)

    # configure optimization
    ITERATIONS = cfg['iterations']
    LR = cfg['lr']
    START_LR_DECAY = cfg['start_lr_decay_iteration']
    loss_weights = cfg['loss_weights']
    loss_tracker = LossTracker(loss_weights[0].keys())
    
    body_optimizer = torch.optim.Adam(body_model_params.parameters(), lr=LR)
    chamfer_distance = ChamferDistance()
    if cfg['body_model'] == 'smpl': # The prior is meaningless for SMPL
        prior = MaxMixturePrior(prior_folder=cfg["prior_path"], num_gaussians=8)
        prior = prior.to(DEVICE)
    if cfg['body_model'] == 'skel': # We need a reference scan to align the SKEL model
        reference_smpl_vertices, _ = load_scan(cfg["reference_smpl"])
        reference_smpl_vertices = torch.from_numpy(reference_smpl_vertices).type(DEFAULT_DTYPE).unsqueeze(0).to(DEVICE)
        prior = SMPL_to_SKEL_vert_distance(reference_smpl_vertices)
        prior = prior.to(DEVICE)

    # Ignore vertices that are not part of the body scan
    ignore_segments = cfg["ignore_segments"]
    ignore_verts = []
    if ignore_segments or type(ignore_segments) is list:
        import json
        with open("SMPL-Fitting/smpl_vert_segmentation.json", 'r') as f:
            vert_segmentation = json.load(f)
        for key in ignore_segments:
            ignore_verts.extend(vert_segmentation[key])
        mask = torch.ones(6890, dtype=torch.bool)
        mask[ignore_verts] = False
        print(sum(mask), 'vertices are used for fitting. Warning: The total number of vertices is hardcoded: 6890.')   
    else:
        mask = torch.ones(6890, dtype=torch.bool)

    if VOLUME_TARGET:
        loss_tracker = LossTracker(list(loss_weights[0].keys())+ ['volume'])
        import volume_utils
        volume_getter = volume_utils.VolumeGetter(ignore_verts)

    if VISUALIZE:
        fig = set_init_plot(input_vertices[0].detach().cpu(), 
                            body_model.verts_t_pose.detach().cpu(), 
                            title=f"Fitting ({input_name}) - initial setup")
        send_to_socket(fig, socket, SOCKET_TYPE)
    print(f"Starting fitting for {input_name} -----------------")


    iterator = tqdm(range(ITERATIONS))
    for i in iterator:

        if VERBOSE: print(f"iteration {i}","red")

        if i in loss_weights.keys():
            if VERBOSE: print(f"\tChanging loss weights","red")
            data_loss_weight = loss_weights[i]['data']
            landmark_loss_weight = loss_weights[i]['landmark']
            prior_loss_weight = loss_weights[i]['prior']
            beta_loss_weight = loss_weights[i]['beta']

        if VERBOSE: print_loss_weights(data_loss_weight,landmark_loss_weight,
                                        prior_loss_weight,beta_loss_weight,
                                        "loss weights:")

        # forward
        pose, beta, trans, scale = body_model_params.forward()
        if VERBOSE: print_params(pose,beta,trans,scale)

        body_model_verts = body_model.deform_verts(pose.to(DEVICE),
                                                   beta.to(DEVICE),
                                                   trans.to(DEVICE),
                                                   scale.to(DEVICE))

        # compute losses
        dist1, dist2, _ , _ = chamfer_distance(body_model_verts[mask].unsqueeze(0), input_vertices)
        data_loss = (torch.mean(dist1)) + (torch.mean(dist2))
        data_loss_weighted = data_loss_weight * data_loss
        if len(body_model_landmark_inds) > 0:
            landmark_loss = summed_L2(body_model_verts[body_model_landmark_inds,:], input_landmarks)
            landmark_loss_weighted = landmark_loss_weight * landmark_loss
        else:
            landmark_loss = torch.tensor(0.0).to(DEVICE)
            landmark_loss_weighted = torch.tensor(0.0).to(DEVICE)
        if cfg['body_model'] != 'skel':
            prior_loss = prior.forward(pose[:, 3:], beta)
        else:
            prior_loss = prior.forward(body_model_verts)
        prior_loss_weighted = prior_loss_weight * prior_loss
        beta_loss = (beta**2).mean()
        beta_loss_weighted = beta_loss_weight * beta_loss
        loss = data_loss_weighted + landmark_loss_weighted + prior_loss_weighted + beta_loss_weighted

        if VOLUME_TARGET and (i>300):
            current_volume = volume_getter.get_volume(body_model_verts)
            volume_loss = (current_volume - VOLUME_TARGET)**2
            loss += volume_loss * 100
            loss_tracker.update({"data": data_loss_weighted,
                            "landmark": landmark_loss_weighted,
                            "prior": prior_loss_weighted,
                            "beta": beta_loss_weighted,
                            "total": loss,
                            "volume": current_volume})
            
        else:
            loss_tracker.update({"data": data_loss_weighted,
                            "landmark": landmark_loss_weighted,
                            "prior": prior_loss_weighted,
                            "beta": beta_loss_weighted,
                            "total": loss})
        
        if VERBOSE: 
            print_losses(data_loss,landmark_loss,prior_loss,beta_loss,"losses")
            print_losses(data_loss_weighted,landmark_loss_weighted,
                         prior_loss_weighted,beta_loss_weighted,"losses weighted")
        iterator.set_description(f"Loss {loss.item():.4f}")

        # optimize
        body_optimizer.zero_grad()
        loss.backward()
        body_optimizer.step()

        if i >= START_LR_DECAY:
            if (i - START_LR_DECAY)%10:
                LR = LR*0.5 
            for param_group in body_optimizer.param_groups:
                param_group['lr'] = LR*(ITERATIONS-i)/ITERATIONS
            if VERBOSE: print((f"\tlr: {param_group['lr']}","yellow"))

        if VISUALIZE and (i in VISUALIZE_STEPS):
            new_title = f"Fitting {input_name} - iteration {i}"
            fig = viz_iteration(fig, body_model_verts.detach().cpu(), i , new_title)
            send_to_socket(fig, socket, SOCKET_TYPE)

            new_title = f"Fitting {input_name} losses - iteration {i}"
            fig_losses = viz_error_curves(loss_tracker.losses, loss_weights, 
                                          new_title, VISUALIZE_LOGSCALE)
            send_to_socket(fig_losses, socket, SOCKET_TYPE)

        DEBUG = False
        if DEBUG:
            import trimesh
            import matplotlib
            matplotlib.use("TkAgg")
            trimesh = trimesh.Trimesh(vertices=body_model_verts.detach().cpu().numpy(), faces=np.array(body_model.faces))
            # Make results directory
            os.makedirs(f"results/tmp", exist_ok=True)
            trimesh.export(f"results/tmp/{input_name}_iter_{i}.obj")
            # Create a plt plot that is dynamically updated
            import matplotlib.pyplot as plt
            plt.ion()
            if i==0:
                fig, ax = plt.subplots()

                line1, = ax.plot([], [], 'r-', label = 'Data Loss')
                line2, = ax.plot([], [], 'g-', label = 'Landmark Loss')
                line3, = ax.plot([], [], 'b-', label = 'Prior Loss')
                line4, = ax.plot([], [], 'y-', label = 'Beta Loss')

                ax.legend()
            # y axis log scale
            ax.set_yscale('log')
            ax.set_xlim(0, i)
            
            line1.set_xdata(range(i+1))
            line1.set_ydata(loss_tracker.losses['data'])
            line2.set_xdata(range(i+1))
            line2.set_ydata(loss_tracker.losses['landmark'])
            line3.set_xdata(range(i+1))
            line3.set_ydata(loss_tracker.losses['prior'])
            line4.set_xdata(range(i+1))
            line4.set_ydata(loss_tracker.losses['beta'])
                # Optionally, adjust the axes limits
            ax.relim()           # Recalculate limits based on new data
            ax.autoscale_view()  # Autoscale

            # Draw the updated figure
            fig.canvas.draw()
            fig.canvas.flush_events()

    if DEBUG:
        plt.ioff()
        plt.show()


    if VISUALIZE:
        fig = viz_final_fit(input_vertices[0].detach().cpu(), 
                          body_model_verts.detach().cpu(),
                          input_faces,
                          title=f"Fitting {input_name} - final fit")
        send_to_socket(fig, socket, SOCKET_TYPE)

    with torch.no_grad():
        pose, beta, trans, scale = body_model_params.forward()
        body_model_verts = body_model.deform_verts(pose, beta, trans, scale)
        fitted_body_model_verts = body_model_verts.detach().cpu().data.numpy()
        fitted_pose = pose.detach().cpu().numpy()
        fitted_shape = beta.detach().cpu().numpy()
        trans = trans.detach().cpu().numpy()
        scale = scale.detach().cpu().numpy()

        save_to = os.path.join(SAVE_PATH,f"{input_name}.npz")
        np.savez(save_to, 
                 body_model = body_model.body_model_name,
                 vertices=fitted_body_model_verts, 
                 pose=fitted_pose, 
                 shape=fitted_shape, 
                 trans=trans, 
                 scale=scale, 
                 name=input_name,
                 scan_index=input_index)


def fit_body_model_onto_dataset(cfg: dict):
    
    # get dataset
    dataset_name = cfg["dataset_name"]
    cfg_dataset = cfg[dataset_name]
    cfg_dataset["use_landmarks"] = cfg["use_landmarks"]
    dataset = eval(cfg["dataset_name"])(**cfg_dataset)

    # if continuing fitting process, get fitted and skipped scans
    fitted_scans = get_already_fitted_scan_names(cfg)
    skipped_scans = get_skipped_scan_names(cfg)

    for i in range(len(dataset)):
        input_example = dataset[i]
        scan_name = input_example["name"]
        print(f"Fitting scan {scan_name} -----------------")

        if (scan_name in fitted_scans) or \
            (scan_name in skipped_scans):
            continue
        
        process_scan = check_scan_prequisites_fit_bm(input_example)
        if process_scan:
            input_example["scan_index"] = i
            fit_body_model(input_example, cfg)
        else:
            skipped_scans.append(input_example["name"])
            to_txt(skipped_scans, cfg["save_path"], "skipped_scans.txt")
    print(f"Fitting for {dataset_name} dataset completed!")


def fit_body_model_onto_scan(cfg: dict):
    scan_name = cfg["scan_path"].split("/")[-1].split(".")[0]
    scan_vertices, scan_faces = load_scan(cfg["scan_path"])
    scan_vertices = scan_vertices / cfg["scale_scan"]

    landmarks = load_landmarks(cfg["landmark_path"],
                              cfg["use_landmarks"],
                              scan_vertices)
    landmarks = {lm_name: (np.array(lm_coord) / cfg["scale_landmarks"]).tolist()
                 for lm_name, lm_coord in landmarks.items()}

    input_example = {"name": scan_name,
                    "vertices": scan_vertices,
                    "faces": scan_faces,
                    "landmarks": landmarks,
                    }

    process_scan = check_scan_prequisites_fit_bm(input_example)
    if process_scan:
        input_example["scan_index"] = 0
        fit_body_model(input_example, cfg)
   


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        help="Subparsers determine the fitting mode: onto_scan or onto_dataset.")
    
    parser_scan = subparsers.add_parser('onto_scan')
    parser_scan.add_argument("--scan_path", type=str, required=True)
    parser_scan.add_argument("--scale_scan", type=float, default=1.0, 
                             help="Scale (divide) the scan vertices by this factor.")
    parser_scan.add_argument("--landmark_path", type=str, required=True)
    parser_scan.add_argument("--scale_landmarks", type=float, default=1.0,
                             help="Scale (divide) the scan landmarks by this factor.")
    parser_scan.set_defaults(func=fit_body_model_onto_scan)

    parser_dataset = subparsers.add_parser('onto_dataset')
    parser_dataset.add_argument("-D","--dataset_name", type=str, required=True)
    parser_dataset.add_argument("-C", "--continue_run", type=str, default=None,
        help="Path to results folder of YYYY_MM_DD_HH_MM_SS format to continue fitting.")
    parser_dataset.set_defaults(func=fit_body_model_onto_dataset)
    
    args = parser.parse_args()


    # load configs
    cfg = load_config()
    cfg_optimization = cfg["fit_body_model_optimization"]
    cfg_datasets = cfg["datasets"]
    cfg_paths = cfg["paths"]
    cfg_general = cfg["general"]
    cfg_web_visualization = cfg["web_visualization"]
    cfg_loss_weights = load_loss_weights_config(
            which_strategy="fit_bm_loss_weight_strategy",
            which_option=cfg_optimization["loss_weight_option"])
    cfg_loss_weights = initialize_fit_bm_loss_weights(cfg_loss_weights)

    # merge configs
    cfg = {}
    cfg.update(cfg_optimization)
    cfg.update(cfg_datasets)
    cfg.update(cfg_paths)
    cfg.update(cfg_general)
    cfg.update(cfg_web_visualization)
    cfg.update(vars(args))
    cfg["loss_weights"] = cfg_loss_weights
    cfg["continue_run"] = cfg["continue_run"] if "continue_run" in cfg.keys() else None

    # process configs
    cfg["save_path"] = create_results_directory(cfg["save_path"], 
                                                cfg["continue_run"])
    cfg = process_default_dtype(cfg)
    cfg = process_visualize_steps(cfg)
    cfg = process_landmarks(cfg)
    cfg = process_body_model_path(cfg)

    # check if landmarks to use are defined
    assert cfg["use_landmarks"] != [], "Please define landmarks to use in config file!"

    # save configs into results dir
    save_configs(cfg)

    # wrapped in a try-except to make sure that the 
    # web visualization socket is closed properly
    try:
        args.func(cfg)
    except (Exception,KeyboardInterrupt) as e:
        print(e)

    if cfg["visualize"]:
        cleanup(cfg["visualize"], cfg["socket"])