
import plotly.graph_objects as go
import plotly.express as px
colors = px.colors.qualitative.Alphabet
import numpy as np
import torch
import argparse
import smplx
import os
from glob import glob
import re

import landmarks
from utils import load_config, load_landmarks, load_scan

######################################################
#               FITTING VISUALIZATION                #
######################################################

def set_init_plot(input_scan: torch.tensor,
                  initial_body_model: torch.tensor,
                  title: str = "Optimization", 
                  subsample_input_scan_verts: int = 100):
    """
    Plot the initial plotly figure for the optimization.
    Plotting the input scan as point cloud
    and the initial body model template as point cloud.

    :param input_scan: (torch.tensor) of dim (N,3)
    :param initial_body_model: (torch.tensor) of dim (M,3)
    :param title: (str) title of the plot
    :param subsample_input_scan_verts: (int) plot every 
                                     k-th vertex

    :return fig: updated plotly figure
    """
    metadata = dict(dash_plot="3d-plot")

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(x = input_scan[::subsample_input_scan_verts,0], 
                                y = input_scan[::subsample_input_scan_verts,1], 
                                z = input_scan[::subsample_input_scan_verts,2], 
                    mode='markers',
                    marker=dict(
                        color='lightpink',
                        size=3,
                        line=dict(
                            color='black',
                            width=1)
                            ),
                    name='input scan',
                    meta=metadata))

    fig.add_trace(go.Scatter3d(x = initial_body_model[:,0], 
                                y = initial_body_model[:,1], 
                                z = initial_body_model[:,2], 
                    mode='markers',
                    marker=dict(
                        color='rgba(0, 0, 100, 0.0)',
                        size=3,
                        line=dict(
                            color='black',
                            width=1
                        )),
                    name='started from',
                    meta=metadata))

    fig.update_layout(scene_aspectmode='data',
                    width=900, height=700,
                    title=title)
    
    return fig


def viz_iteration(fig,pc,iteration,title=None):
    """
    Plot the points of the optimized body model 
    for iteration

    :param fig: (plotly) figure
    :param pc: (torch.tensor) (N,3)
    :param iteration: (int) current iteration
    :param title: (str) title of plot

    :return fig: updated plotly figure
    """

    metadata = dict(dash_plot="3d-plot")

    if title:
        fig.update_layout(title=title)
    else:
        fig.update_layout(title=f"Fitting iteration {iteration}")

    name = f"Iter {iteration}"
    fig.add_trace(go.Scatter3d(x = pc[:,0], 
                               y = pc[:,1], 
                               z = pc[:,2], 
                mode='markers',
                marker=dict(
                        color='rgba(0, 206, 250, 0.0)',
                        size=3,
                        line=dict(
                            color='blue',
                            width=1
                        )),
                name=name,
                meta=metadata
                ))
    return fig


def viz_error_curves(losses, loss_weight_strategy, 
                     title="Error curves",logscale=False):
    """
    Visualize error curves (losses) for the optimization. 
    Vertical lines are added when weights of losses are changed.


    :param losses: (dict) of losses, e.g. {"loss1": [1,2,3], "loss2": [4,5,6]}
    :param loss_weight_strategy: (dict) of loss weight strategies, 
                                   e.g. {1: {"loss1": 0.5, "loss2": 0.5}, 
                                         2: {"loss1": 0.3, "loss2": 0.7}}
                                 where keys are the iteration number
                                 defined in configs/loss_weight_configs.yaml
    :param title: (str) title of the plot
    :param logscale: (bool) whether to plot the curves in logscale

    :return curves: plotly figure
    """

    curves = go.Figure()
    metadata = dict(dash_plot="error-curves")

    # plot loss curves
    for loss_name,loss_values in losses.items():
        if loss_values: # if list not empty, plot curve
            N = len(loss_values)
            if logscale:
                y = np.log(np.array(loss_values))
            else:
                y = np.array(loss_values)

            curves.add_trace(go.Scatter(x=np.arange(N),
                                        y=y,
                                    mode='lines+markers',
                                    name=loss_name,
                                    meta=metadata))
            
    # plot vertical lines indicating change of loss weight
    # plot only to current iteration
    current_iteration = len(losses[list(losses.keys())[0]])
    for nr_iter, loss_weights in loss_weight_strategy.items():
        if nr_iter > current_iteration:
            continue
        names = [f"{ln}:{lw:.2f}" for ln,lw in loss_weights.items()]
        names_str = ", ".join(names)
        names_str = f"loss weights: {names_str}"
        curves.add_vline(x=nr_iter, 
                         line_width=2, line_dash="dash", 
                         line_color="red",
                         annotation_text=names_str,
                         annotation_font_color="red",
                        #  meta=metadata
                         )
            
    curves.update_layout(scene_aspectmode='data',
                         hovermode='x',
                         width=900, height=700,
                         title=title,
                         annotations=[dict(textangle=-90,
                                           y=0.1,
                                           yref="paper",
                                           yanchor="bottom")])
            
    return curves


def viz_final_fit(input_scan_verts: torch.tensor,
                fitted_body_model: torch.tensor,
                input_scan_faces: torch.tensor = None,
                title: str = "Final fit",
                subsample_rate: int = 20):
    """
    Plot the final fit figure for the optimization.
    Plotting the input scan as mesh if faces are given,
    else plotting as point cloud.
    Plotting the fitted body model as point cloud.

    :param input_scan_verts: (torch.tensor) dim (N,3)
    :param fitted_body_model: (torch.tensor) dim (M,3)
    :param input_scan_faces: (torch.tensor) dim (F,3)
    :param title: (str) title of the plot
    :param subsample_rate: (int) plot every subsample_rate-th 
                            point of the input scan

    :return fig: plotly figure
    """

    metadata = dict(dash_plot="final-fit")

    fig = go.Figure()
    if not isinstance(input_scan_faces,type(None)):
        fig.add_trace(go.Mesh3d(
                            x=input_scan_verts[:,0], 
                            y=input_scan_verts[:,1], 
                            z=input_scan_verts[:,2], 
                            i=input_scan_faces[:,0], 
                            j=input_scan_faces[:,1], 
                            k=input_scan_faces[:,2], 
                            color='lightpink', 
                            opacity=0.9,
                            meta=metadata
                    ))
    else:
        fig.add_trace(go.Scatter3d(
                        x = input_scan_verts[::subsample_rate,0], 
                        y = input_scan_verts[::subsample_rate,1], 
                        z = input_scan_verts[::subsample_rate,2], 
                        mode='markers',
                        marker=dict(
                            color='lightpink',
                            size=3,
                            line=dict(
                                color='black',
                                width=1)
                                ),
                        name='input scan',
                        meta=metadata
                        ))

    fig.add_trace(go.Scatter3d(x = fitted_body_model[:,0], 
                                y = fitted_body_model[:,1], 
                                z = fitted_body_model[:,2], 
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=3,
                        line=dict(
                            color='black',
                            width=1
                        )),
                    name='Fitted template',
                    meta=metadata
                    ))

    fig.update_layout(scene_aspectmode='data',
                    width=900, height=700,
                    title=title)
    
    return fig


######################################################
#               VIZ FUNCS                            #
######################################################

def create_wireframe_plot(verts: np.ndarray,faces: np.ndarray):
        '''
        Given vertices and faces, creates a wireframe of plotly segments.
        Used for visualizing the wireframe.
        
        :param verts: np.array (N,3) of vertices
        :param faces: np.array (F,3) of faces connecting the verts
        '''
        i=faces[:,0]
        j=faces[:,1]
        k=faces[:,2]

        triangles = np.vstack((i,j,k)).T

        x=verts[:,0]
        y=verts[:,1]
        z=verts[:,2]

        vertices = np.vstack((x,y,z)).T
        tri_points = vertices[triangles]

        #extract the lists of x, y, z coordinates of the triangle 
        # vertices and connect them by a "line" by adding None
        # this is a plotly convention for plotting segments
        Xe = []
        Ye = []
        Ze = []
        for T in tri_points:
            Xe.extend([T[k%3][0] for k in range(4)]+[ None])
            Ye.extend([T[k%3][1] for k in range(4)]+[ None])
            Ze.extend([T[k%3][2] for k in range(4)]+[ None])

        # return Xe, Ye, Ze 
        wireframe = go.Scatter3d(
                        x=Xe,
                        y=Ye,
                        z=Ze,
                        mode='lines',
                        name='wireframe',
                        line=dict(color= 'rgb(70,70,70)', width=1)
                        )
        return wireframe

def visualize_smpl_landmarks(**kwargs):
    cfg = load_config()

    # get body model
    body_models_path = cfg["paths"]["body_models_path"]
    smpl_body_model_path = os.path.join(body_models_path,
                                        "smpl/SMPL_NEUTRAL.pkl")
    body_model = smplx.create(smpl_body_model_path, 
                            model_type="SMPL",
                            gender="NEUTRAL", 
                            use_face_contour=False,
                            ext='pkl')
    
    body_model_vertices = body_model.v_template
    body_mdoel_faces = body_model.faces
    
    # get landmarks
    lm = landmarks.SMPL_INDEX_LANDMARKS
    if not isinstance(kwargs["select_landmarks_to_viz"],type(None)):
        lm = {k:v for k,v in lm.items() 
              if k in kwargs["select_landmarks_to_viz"]}

    # plot body and landmarks
    fig = go.Figure()

    ## plot body
    plot_body = go.Mesh3d(
                        x=body_model_vertices[:,0],
                        y=body_model_vertices[:,1],
                        z=body_model_vertices[:,2],
                        #facecolor=face_colors,
                        color = "lightpink",
                        i=body_mdoel_faces[:,0],
                        j=body_mdoel_faces[:,1],
                        k=body_mdoel_faces[:,2],
                        name='smpl body model',
                        hovertemplate ='<i>vert ind:</i>: %{text}',
                        text = list(range(body_model_vertices.shape[0])),
                        showscale=True,
                        opacity=0.7
                        )
    fig.add_trace(plot_body)

    ## plot landmarks
    n_colors = len(lm.keys())
    lm_colors = px.colors.sample_colorscale("turbo", [n/(n_colors -1) for n in range(n_colors)])

    for i, (lm_name, lm_ind) in enumerate(lm.items()):
        fig.add_trace(go.Scatter3d(x = [body_model_vertices[lm_ind,0]], 
                                    y =[body_model_vertices[lm_ind,1]], 
                                    z = [body_model_vertices[lm_ind,2]], 
                        mode='markers',
                        marker=dict(
                            color=lm_colors[i],
                            size=8,
                            line=dict(
                                color='black',
                                width=1)
                                ),
                        name=lm_name))

    fig.update_layout(scene_aspectmode='data',
                        width=1000, height=700,
                        title="SMPL + landmarks",
                        )
    
    fig.show()

def visualize_scan_landmarks(scan_path,landmark_path, **kwargs):

    verts, faces = load_scan(scan_path)
    landmarks = load_landmarks(landmark_path)

    fig = go.Figure()

    ## plot body
    if isinstance(faces,type(None)):
        plot_body = go.Scatter3d(x = verts[:,0], 
                                 y =verts[:,1], 
                                 z = verts[:,2], 
                        mode='markers',
                        marker=dict(
                            color="lightpink",
                            size=8,
                            line=dict(
                                color='black',
                                width=1)
                                ),
                        name="Scan")
    else:
        plot_body = go.Mesh3d(
                            x=verts[:,0],
                            y=verts[:,1],
                            z=verts[:,2],
                            color = "lightpink",
                            i=faces[:,0],
                            j=faces[:,1],
                            k=faces[:,2],
                            name='Scan',
                            showscale=True,
                            opacity=0.7
                            )
    fig.add_trace(plot_body)

    ## plot landmarks
    n_colors = len(landmarks.keys())
    lm_colors = px.colors.sample_colorscale("turbo", [n/(n_colors -1) for n in range(n_colors)])

    for i, (lm_name, lm_coord) in enumerate(landmarks.items()):
        fig.add_trace(go.Scatter3d(x = [lm_coord[0]], 
                                   y =[lm_coord[1]], 
                                   z = [lm_coord[2]], 
                        mode='markers',
                        marker=dict(
                            color=lm_colors[i],
                            size=8,
                            line=dict(
                                color='black',
                                width=1)
                                ),
                        name=lm_name))
    
    fig.update_layout(scene_aspectmode='data',
                        width=1000, height=700,
                        title="Scan + landmarks",
                        )
    
    fig.show()

def visualize_fitting(scan_path, fit_paths, fit_nicknames=None, scale_scan=None, 
                      return_fig=False, visualize_mesh_texture=False, **kwargs):
     
    if visualize_mesh_texture:
        verts, faces, verts_colors = load_scan(scan_path, 
                                     return_vertex_colors=visualize_mesh_texture)
    else:
        verts, faces = load_scan(scan_path)
        verts_colors = None

    if scale_scan:
       verts = verts / scale_scan 

    fig = go.Figure()

    ## plot scan
    if isinstance(faces,type(None)):
        plot_body = go.Scatter3d(x = verts[:,0], 
                                 y =verts[:,1], 
                                 z = verts[:,2], 
                                 mode='markers',
                                 marker=dict(
                                    color="lightpink",
                                    size=8,
                                    line=dict(
                                        color='black',
                                        width=1)
                                        ),
                                 name="Scan")
    else:
        plot_body = go.Mesh3d(
                            x=verts[:,0],
                            y=verts[:,1],
                            z=verts[:,2],
                            color = "lightpink",
                            i=faces[:,0],
                            j=faces[:,1],
                            k=faces[:,2],
                            name='Scan',
                            showscale=True,
                            opacity=0.7,
                            vertexcolor=verts_colors,
                            flatshading=True
                            )
        
        ## add wireframe of scan
        wireframe_plot = create_wireframe_plot(verts,faces)
        fig.add_trace(wireframe_plot)

    fig.add_trace(plot_body)


    if not isinstance(fit_nicknames,type(None)):
        if len(fit_nicknames) != len(fit_paths):
            print("Number of fit_nicknames does not match number of fit_paths.")
            print("Using names extracted from paths")

    experiment_names = []
    fitted_names = []
    n_colors = len(fit_paths)
    # want colors from 0.1 to 1 because below 0.1 is black
    colors = px.colors.sample_colorscale("turbo", 
                                         list(np.linspace(0.1, 1, n_colors))
                                    # [0.1 + n/n_colors for n in range(n_colors)]
                                    )

    for i, fit_path in enumerate(fit_paths):

        # find standard experiment name
        experiment_name = re.search(r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}', fit_path)
        if not isinstance(experiment_name,type(None)):
            experiment_name = experiment_name.group(0)
        else:
            experiment_name = os.path.basename(os.path.dirname(fit_path))
        experiment_names.append(experiment_name)

        if not isinstance(fit_nicknames,type(None)):
            viz_name = fit_nicknames[i]
        else:
            viz_name = experiment_name

        fitted_data = np.load(fit_path)
        fitted_verts = fitted_data["vertices"]
        fitted_name = str(fitted_data["name"])
        fitted_names.append(fitted_name)

        # plot fitted_verts
        plot_fitted = go.Scatter3d(x = fitted_verts[:,0],
                                y = fitted_verts[:,1],
                                z = fitted_verts[:,2],
                            mode='markers',
                            marker=dict(
                                color=colors[i],
                                size=8,
                                line=dict(
                                    color='black',
                                    width=1)
                                    ),
                            name=viz_name)
        fig.add_trace(plot_fitted)

    if np.unique(fitted_names).shape[0] > 1:
        raise ValueError("Visualizing fits for wrong scan.")

    viz_title = f"Viz scan {fitted_name} and fitted bm from " + \
                f"experiments {experiment_names}"
    fig.update_layout(scene_aspectmode='data',
                    width=1000, height=1000,
                    title=viz_title,
                        )
    if return_fig:
        return fig
    else:
        fig.show()


def visualize_pve(verts, vert_errors, faces, name=""):
    """
    Visualize the PVE errors as a color map on the fitted body model.
    :param verts: np.array (N,3)
    :param vert_errors: np.array (N,)
    :param faces: np.array (F,3)
    :param name: name of the mesh
    """

    fig = go.Figure()

    vert_errors_cm = vert_errors*100

    plot_errors = go.Mesh3d(
        x=verts[:,0],
        y=verts[:,1],
        z=verts[:,2],
        colorbar_title='pve (cm)',
        colorscale="Plasma",
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity=vert_errors_cm,
        # i, j and k give the vertices of triangles
        # here we represent the 4 triangles of the tetrahedron surface
        i=faces[:,0],
        j=faces[:,1],
        k=faces[:,2],
        # when hover show error
        hovertemplate ='<i>pve:</i>: %{text:.4f}cm',
        text = vert_errors_cm,
        name='Fitted body model',
        showscale=True
    )

    fig.add_trace(plot_errors)

    fig.update_layout(scene_aspectmode='data',
                    width=1000, height=700,
                    title=f"{name} pve errors",
                        )

    fig.show()
                             

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="subparsers")

    parser_viz_smpl_landmarks = subparsers.add_parser('visualize_smpl_landmarks')
    parser_viz_smpl_landmarks.add_argument('--select_landmarks_to_viz', nargs='+', required=False,
                                    help='Select subset of landmarks.', 
                                    default=None)
    parser_viz_smpl_landmarks.set_defaults(func=visualize_smpl_landmarks)

    parser_viz_scan_landmarks = subparsers.add_parser('visualize_scan_landmarks')
    parser_viz_scan_landmarks.add_argument("-S", "--scan_path", type=str, required=True)
    parser_viz_scan_landmarks.add_argument("-L", "--landmark_path", type=str, required=True)
    parser_viz_scan_landmarks.set_defaults(func=visualize_scan_landmarks)

    parser_viz_fitting = subparsers.add_parser('visualize_fitting')
    parser_viz_fitting.add_argument("-S", "--scan_path", type=str, required=True)
    parser_viz_fitting.add_argument("-F", "--fit_paths", required=True, 
                                    nargs='+', default=[],
                                    help="One or multiple paths to the fitted npz file \
                                          you want to visualize.")
    parser_viz_fitting.add_argument("--fit_nicknames", required=False, 
                                    nargs='+', default=[],
                                    help="More meaningful names for each experiment in the \
                                          fit_paths you are visualizing.")
    parser_viz_fitting.add_argument("--visualize_mesh_texture", action="store_true")
    parser_viz_fitting.add_argument("--scale_scan", type=float, required=False, default=1.0)
    parser_viz_fitting.set_defaults(func=visualize_fitting)

    
    args = parser.parse_args()

    args.func(**vars(args))