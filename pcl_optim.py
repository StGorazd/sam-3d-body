import json
import os

import cv2
import numpy as np
import pytorch3d
from pytorch3d.io import load_ply
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.loss import point_mesh_face_distance

import scipy
import torch
import rerun as rr
import trimesh
from tqdm import tqdm

from utils.image import load_image, is_image
from utils.measure import get_measurements
from utils.pointcloud import get_moge_pointcloud, get_scaled_pointcloud

class SinglePoseMHR(torch.nn.Module):
    def __init__(self, fname, subsample=5000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mhr_model = torch.jit.load("assets/mhr_model.pt")
        self.register_buffer('faces', self.mhr_model.character_torch.mesh.faces.unsqueeze(0))

        rr.init(f"Viewer - Single Pose", spawn=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log("world/XYZ", rr.Arrows3D(vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))

        pcl_mesh = trimesh.load(fname)

        # self.register_buffer('pcl', pcl_verts.unsqueeze(0))
        pcl_mesh.vertices[:, 1] *= -1
        # pcl_mesh.vertices[:, 2] *= -1

        self.pcl = Pointclouds(torch.Tensor(pcl_mesh.vertices).unsqueeze(0) / 1000).cuda()
        self.colors = pcl_mesh.colors

        print("Data loaded")

        self.identity = torch.nn.Parameter(torch.zeros([1, 45]), requires_grad=True)
        self.expr = torch.nn.Parameter(torch.zeros([1, 72]), requires_grad=True)
        self.pose = torch.nn.Parameter(torch.zeros([1, 204]), requires_grad=True)
        self.t = torch.nn.Parameter(torch.zeros([1, 3]), requires_grad=True)

    def forward(self):
        mean_model_vertices, skel_state = (
            self.mhr_model(
                model_parameters=self.pose,
                identity_coeffs=self.identity,
                face_expr_coeffs=self.expr,
            )
        )

        mean_model_vertices /= 100
        mean_model_vertices[:, :, 1] *= -1
        mean_model_vertices[:, :, 2] *= -1
        mean_model_vertices += self.t.reshape(-1, 1, 3)

        meshes = Meshes(mean_model_vertices, self.faces)
        # print("Computin loss")
        pcl_loss = pytorch3d.loss.point_mesh_face_distance(meshes, self.pcl)

        return mean_model_vertices, pcl_loss

    def visualize(self, i, meshes):
        rr.set_time("frame_idx", sequence=i)

        rr.log(f"pointcloud", rr.Points3D(self.pcl[0].cpu().points_packed().numpy(), colors=self.colors))
        # rr.log(f"view-{j}/subsampled_pointcloud", rr.Points3D(self.pcls[j].points_list(),
        #                                                       colors=self.pcls[j].features_list()))

        rr.log(f"mhr_model", rr.Mesh3D(vertex_positions=meshes[0].detach().cpu().numpy(),
                                                 triangle_indices=self.faces[0].cpu(),
                                                 # vertex_normals=meshes[j].vertex_normals,
                                                 vertex_colors=np.array([0, 0, 190])))




if __name__ == '__main__':
    image_dir = 'D:/Research/data/antropo/x1'
    data_dir = 'D:/Research/data/antropo/x1'

    fname = 'D:/Research/data/antropo/pointcloud/body_processed.ply'
    # fnames = ['IMG_9581.jpeg']

    model = SinglePoseMHR(fname)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    pbar = tqdm(range(500))
    for i in pbar:
        optimizer.zero_grad()
        vertices, loss = model()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss: {loss.item():0.6f}")
        if i % 10 == 0:
            model.visualize(i, vertices)