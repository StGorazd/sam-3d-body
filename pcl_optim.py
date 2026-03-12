import json
import os
import sys

import cv2
import numpy as np
import pytorch3d
from pytorch3d.io import load_ply
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.loss import point_mesh_face_distance

import scipy
import torch
import rerun as rr
import trimesh
from timm.models import model_parameters
from tqdm import tqdm
import time
from mhr.mhr import MHR
from pathlib import Path

# from utils.image import load_image, is_image
# from utils.measure import get_measurements
# from utils.pointcloud import get_moge_pointcloud, get_scaled_pointcloud

class SinglePoseMHR(torch.nn.Module):
    def __init__(self, fname, subsample=5000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.mhr_model = torch.jit.load("assets/mhr_model.pt")
        # self.register_buffer('faces', self.mhr_model.character_torch.mesh.faces.unsqueeze(0))
        self.mhr_model = MHR.from_files(
            folder=Path("assets/assets"),
            device=torch.device("cuda"),
            lod=4
        )
        faces = torch.as_tensor(
            self.mhr_model.character.mesh.faces,
            dtype=torch.float64,
            device="cuda"
        )
        self.register_buffer("faces", faces.unsqueeze(0))

        rr.init(f"Viewer - Single Pose", spawn=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log("world/XYZ", rr.Arrows3D(vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))

        pcl_mesh = trimesh.load(fname)

        # self.register_buffer('pcl', pcl_verts.unsqueeze(0))
        pcl_mesh.vertices[:, 1] *= -1
        pcl_mesh.vertices[:, 2] *= -1

        # self.pcl = Pointclouds(torch.Tensor(pcl_mesh.vertices).unsqueeze(0) / 1000).cuda()
        self.pcl = Pointclouds(
            torch.tensor(pcl_mesh.vertices, dtype=torch.float32, device="cuda").unsqueeze(0) / 1000
        )

        self.colors = pcl_mesh.colors

        print("Data loaded")

        self.identity = torch.nn.Parameter(torch.zeros([1, 45]), requires_grad=True)
        # self.expr = torch.nn.Parameter(torch.zeros([1, 72]), requires_grad=True)
        # Excluding face expression from optimizing
        self.expr = self.register_buffer("expr", torch.zeros([1, 72], device="cuda"))
        self.pose = torch.nn.Parameter(torch.zeros([1, 204]), requires_grad=True)
        self.t = torch.nn.Parameter(torch.zeros([1, 3]), requires_grad=True)

        self.register_buffer("finger_pose_mask", self._create_finger_mask())

    def _start_timer(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def _end_timer(self, start_event):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end)  # milliseconds

    def _create_finger_mask(self):
        # https://github.com/facebookresearch/MHR/blob/main/tools/mhr_smpl_conversion/conversion.py#L658
        num_total_params = self.pose.shape[1]

        num_pose_param = int(self.mhr_model.character.parameter_transform.pose_parameters.sum() - 6)
        num_scale_params = int(self.mhr_model.character.parameter_transform.scaling_parameters.sum())

        lbs_parameter_names = self.mhr_model.character.parameter_transform.names[
            6: 6 + num_pose_param + num_scale_params
        ]

        finger_parts = {"index", "middle", "ring", "pinky", "thumb"}

        mask = torch.ones(num_total_params, dtype=torch.float32, device="cuda")

        for i, name in enumerate(lbs_parameter_names[:num_pose_param]):
            if any(part in name.lower() for part in finger_parts):
                mask[6 + i] = 0.0

        return mask.unsqueeze(0)  # shape [1, 204]

    def forward(self):
        # t1 = self._start_timer()
        # mean_model_vertices, skel_state = (
        #     self.mhr_model(
        #         model_parameters=self.pose,
        #         identity_coeffs=self.identity,
        #         face_expr_coeffs=self.expr,
        #     )
        # )
        mean_model_vertices, skel_state = self.mhr_model(
            identity_coeffs=self.identity,
            model_parameters=self.pose,
            face_expr_coeffs=self.expr
        )
        # mhr_time = self._end_timer(t1)

        # t2 = self._start_timer()
        mean_model_vertices /= 100
        mean_model_vertices[:, :, 1] *= -1
        mean_model_vertices[:, :, 2] *= -1
        mean_model_vertices += self.t.reshape(-1, 1, 3)
        # transform_time = self._end_timer(t2)

        # t3 = self._start_timer()
        meshes = Meshes(mean_model_vertices, self.faces)
        # mesh_creation_time = self._end_timer(t3)

        # print("Computin loss")
        # t4 = self._start_timer()
        # pcl_loss = pytorch3d.loss.point_mesh_face_distance(meshes, self.pcl)
        # ======================================================================
        sampled_points = sample_points_from_meshes(meshes, num_samples=40_000)
        pcl_loss, _ = pytorch3d.loss.chamfer_distance(sampled_points, self.pcl.points_padded())
        # loss_time = self._end_timer(t4)

        # total_time = mhr_time + transform_time + mesh_creation_time + loss_time
        # print(f"[ms] MHR={mhr_time:.2f} T={transform_time:.2f} M={mesh_creation_time:.2f} L={loss_time:.2f} Total={total_time:.2f}")

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

    # fname = 'D:/Research/data/antropo/pointcloud/body_processed.ply'
    fname = "pointclouds/body_processed.ply"
    # fnames = ['IMG_9581.jpeg']

    model = SinglePoseMHR(fname)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-5
    )
    pbar = tqdm(range(600))
    for i in pbar:
        optimizer.zero_grad()
        vertices, loss = model()
        loss.backward()
        with torch.no_grad():
            model.pose.grad *= model.finger_pose_mask

        optimizer.step()

        scheduler.step(loss.item())
        currentLr = optimizer.param_groups[0]['lr']

        pbar.set_description(f"Loss: {loss.item():0.6f}, LR: {currentLr:0.6f}")
        if i % 10 == 0:
            model.visualize(i, vertices)