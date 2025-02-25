

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class OptimizationSMPL(torch.nn.Module):
    """
    Class used to optimize SMPL parameters.
    """
    def __init__(self, cfg: dict):
        super(OptimizationSMPL, self).__init__()

        # self.pose = torch.nn.Parameter(torch.zeros(1, 72).to(DEVICE))
        # self.beta = torch.nn.Parameter((torch.zeros(1, 10).to(DEVICE)))
        # self.trans = torch.nn.Parameter(torch.zeros(1, 3).to(DEVICE))
        # self.scale = torch.nn.Parameter(torch.ones(1).to(DEVICE)*1)

        pose = torch.zeros(1, 72).to(DEVICE)
        beta = torch.zeros(1, 10).to(DEVICE)
        trans = torch.zeros(1, 3).to(DEVICE)
        scale = torch.ones(1).to(DEVICE)*1

        if "init_params" in cfg:
            init_params = cfg["init_params"]
            if "pose" in init_params:
                pose = cfg["init_params"]["pose"].to(DEVICE)
            if "shape" in init_params:
                beta = cfg["init_params"]["shape"].to(DEVICE)

            if "trans" in init_params:
                trans = cfg["init_params"]["trans"].to(DEVICE)

            if "scale" in init_params:
                scale = cfg["init_params"]["scale"].to(DEVICE)
        

        if "refine_params" in cfg:
            params_to_refine = cfg["refine_params"]
            if "pose" in params_to_refine:
                self.pose = torch.nn.Parameter(pose)
            else:
                self.pose = pose
            if "shape" in params_to_refine:
                self.beta = torch.nn.Parameter(beta)
            else:
                self.beta = beta
            if "trans" in params_to_refine:
                self.trans = torch.nn.Parameter(trans)
            else:
                self.trans = trans
            if "scale" in params_to_refine:
                self.scale = torch.nn.Parameter(scale)
            else:
                self.scale = scale
        else:
            self.pose = torch.nn.Parameter(pose)
            self.beta = torch.nn.Parameter(beta)
            self.trans = torch.nn.Parameter(trans)
            self.scale = torch.nn.Parameter(scale)

    def forward(self):
        return self.pose, self.beta, self.trans, self.scale
    
class OptimizationSKEL(OptimizationSMPL):
    """
    Class used to optimize SMPL parameters. Different pose parameters than SMPL
    """
    def __init__(self, cfg: dict):
        super(OptimizationSKEL, self).__init__(cfg)
        pose = torch.zeros(1, 46).to(DEVICE)

        if "init_params" in cfg:
            init_params = cfg["init_params"]
            if "pose" in init_params:
                pose = cfg["init_params"]["pose"].to(DEVICE)

        if "refine_params" in cfg:
            params_to_refine = cfg["refine_params"]
            if "pose" in params_to_refine:
                self.pose = torch.nn.Parameter(pose)
        else:
            self.pose = torch.nn.Parameter(pose)



class BodyParameters():
    def __new__(cls, cfg):

        possible_model_types = ["smpl"] #["smpl", "smplx"]
        model_type = cfg["body_model"].lower()

        if model_type == "smpl":
            return OptimizationSMPL(cfg)
        elif model_type == "skel":
            return OptimizationSKEL(cfg)
        #     return OptimizationSMPLX()
        else:
            msg = f"Model type {model_type} not defined. \
                    Possible model types are: {possible_model_types}"
            raise NotImplementedError(msg)
        

