import torch
import sys 
sys.path.append("/home/rzlin/ri94mihu/phd/WP4/SMPL-Fitting/")
from body_models import BodyModel
from utils import load_config, process_body_model_path
import open3d as o3d
import numpy as np

class VolumeGetter:
    def __init__(self, ignore_verts):
        # Add ... to path
        ## Load the body model
        cfg = load_config("/home/rzlin/ri94mihu/phd/WP4/SMPL-Fitting/configs/config.yaml")
        cfg['body_model'] = 'smpl'
        cfg['body_models_path'] = "/home/rzlin/ri94mihu/phd/WP4/SMPL-Fitting/data/body_models"
        cfg = process_body_model_path(cfg)
        model = BodyModel(cfg)
        faces = model.faces
        face_mask = torch.ones(faces.shape[0], dtype=torch.bool)
        for i, face in enumerate(faces):
            for v in face:
                if v in ignore_verts:
                    face_mask[i] = False
                    break
        self.faces = faces
        self.face_mask = face_mask
            

    def get_volume(self, vertices):
        #2.a Get the vertex normals
        faces = np.array(self.faces, dtype=np.int32)
        verts = vertices
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts.detach().cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_triangle_normals()
        normals = torch.from_numpy(np.asarray(mesh.triangle_normals))
        normals = normals.to('cuda')
        #base_vert = torch.mean(verts, axis=0)
        base_vert = torch.tensor([0,0,0.25], dtype=torch.float32, device='cuda')
        # Normals pointing

        dot_product = torch.sum(normals * (verts[faces][:,0]-base_vert), axis=1) #dot_product = torch.sum(normals * (verts[np.array(faces[face_mask], dtype=np.int32)][:,:,0]-base_vert), axis=1)
        dot_product[dot_product > 0] = 1
        dot_product[dot_product < 0] = -1
        face_normals = dot_product[self.face_mask]

        masked_faces = faces[self.face_mask]
        list_of_triangles = verts[masked_faces]

        v1 = torch.cross(list_of_triangles[:,0] - base_vert, list_of_triangles[:,1] - base_vert)
        v2 = list_of_triangles[:,2] - base_vert
        volume = torch.sum(v2*v1, dim=1) / 6
        volume = torch.abs(torch.sum(face_normals * torch.abs(volume)))
        return volume * 1000