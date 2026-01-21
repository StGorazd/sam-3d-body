import json
import os

import cv2
import numpy as np
import pytorch3d
import pytorch3d
from pytorch3d.renderer import RasterizationSettings, MeshRenderer, MeshRasterizer, SoftSilhouetteShader, BlendParams, \
    FoVPerspectiveCameras
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.loss import point_mesh_face_distance

import scipy
import torch
import rerun as rr
import trimesh
from tqdm import tqdm

from utils.image import load_image, is_image
from utils.pointcloud import get_moge_pointcloud, get_scaled_pointcloud

class MultiViewMultiPoseMHR(torch.nn.Module):
    def __init__(self, image_dir, data_dir, fnames, subsample=5000, mask_scale_factor=0.125, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mhr_model = torch.jit.load("assets/mhr_model.pt")
        self.register_buffer('faces', self.mhr_model.character_torch.mesh.faces)
        self.n_views = len(fnames)
        self.subsample = subsample
        self.mask_scale_factor = mask_scale_factor

        rr.init(f"Viewer", spawn=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log("world/XYZ", rr.Arrows3D(vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))

        pose_param_list = []
        identity_params_list = []
        expr_params_list = []
        t_list = []
        self.fov_list = []

        self.point_list = []
        self.color_list = []
        self.mask_list = []

        print("Loading detections")
        for fname in fnames:
            print(fname)
            image_name = fname.split('.')[0]
            image = load_image(os.path.join(image_dir, fname))
            mask = load_image(os.path.join(data_dir, f'{image_name}_mask.jpg'))[:, :, 0] / 255
            mask_pcl = cv2.erode(mask, np.ones((10, 10), np.uint8), iterations=1) > 0.5

            moge_depth = np.load(os.path.join(data_dir, f'{image_name}_depth.npy'))

            with open(os.path.join(data_dir, f'{image_name}.json'), 'r') as f:
                json_data = json.load(f)

            f = json_data['focal_length']
            fov = 2 * np.rad2deg(np.arctan(mask.shape[1] / (2 * f)))
            self.fov_list.append(fov)
            K = np.array([[f, 0, image.shape[1] / 2], [0, f, image.shape[0] / 2], [0, 0, 1]])
            # points = get_moge_pointcloud(moge_depth, K)
            points = get_scaled_pointcloud(image, moge_depth, K)

            masked_points = points.reshape(-1, 3)[mask_pcl.ravel()]
            masked_colors = image[:, :, ::-1].reshape(-1, 3)[mask_pcl.ravel()]
            t = np.array(json_data['pred_cam_t'])

            pose_param_list.append(json_data['mhr_model_params'])
            identity_params_list.append(json_data['shape_params'])
            expr_params_list.append(json_data['expr_params'])
            t_list.append(t)
            self.point_list.append(masked_points)
            self.color_list.append(masked_colors)
            self.mask_list.append(mask)

        print("Data loaded")

        self.identity = torch.nn.Parameter(torch.from_numpy(np.mean(np.array(identity_params_list), axis=0).astype(np.float32)), requires_grad=True)
        self.expr = torch.nn.Parameter(torch.from_numpy(np.mean(np.array(expr_params_list), axis=0).astype(np.float32)), requires_grad=True)
        self.poses = torch.nn.Parameter(torch.from_numpy(np.array(pose_param_list, dtype=np.float32)), requires_grad=True)
        self.ts = torch.nn.Parameter(torch.from_numpy(np.array(t_list, dtype=np.float32)), requires_grad=True)

        self.pcls = Pointclouds([torch.from_numpy(x.astype(np.float32)) for x in self.point_list],
                                [torch.from_numpy(x.astype(np.float32)) for x in self.color_list]).subsample(self.subsample).cuda()

        for i in range(self.n_views):
            self.mask_list[i] = cv2.resize(self.mask_list[i], None, fx=0.25, fy=0.25)
        self.register_buffer('masks', torch.from_numpy(np.array(self.mask_list).astype(np.float32)))

        self.cameras = FoVPerspectiveCameras(fov=self.fov_list, device='cuda')
        blend_params = BlendParams(sigma=1e-5, gamma=1e-6)
        raster_settings = RasterizationSettings(
            image_size=(self.mask_list[0].shape[0], self.mask_list[0].shape[1]),
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=64,
            bin_size=0
        )

        self.R = torch.diag(torch.tensor([-1, -1, 1], dtype=torch.float32)).unsqueeze(0).expand(self.n_views, -1, -1).cuda()

        # Create a silhouette mesh renderer by composing a rasterizer and a shader.
        self.silhouette_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=self.cameras,
                                                                          raster_settings=raster_settings),
                                                shader=SoftSilhouetteShader(blend_params=blend_params))

    def forward(self):
        mean_model_vertices, skel_state = (
            self.mhr_model(
                model_parameters=self.poses,
                identity_coeffs=self.identity.unsqueeze(0).expand(self.n_views, -1),
                face_expr_coeffs=self.expr.unsqueeze(0).expand(self.n_views, -1),
            )
        )

        mean_model_vertices /= 100
        mean_model_vertices[:, :, 1] *= -1
        mean_model_vertices[:, :, 2] *= -1
        mean_model_vertices += self.ts.reshape(-1, 1, 3)

        meshes = Meshes(mean_model_vertices, self.faces.unsqueeze(0).expand(self.n_views, -1, -1))
        # print("Computin loss")
        pcl_loss = pytorch3d.loss.point_mesh_face_distance(meshes, self.pcls)

        rendered_masks = self.silhouette_renderer(meshes, R=self.R, T=torch.zeros(self.n_views, 3).cuda())[..., 3]
        mask_loss = torch.mean((rendered_masks - self.masks) ** 2)

        return mean_model_vertices, rendered_masks, pcl_loss, mask_loss

    def visualize(self, i, meshes, rendered_masks):
        rr.set_time("frame_idx", sequence=i)

        for j in range(self.n_views):
            rr.log(f"view-{j}/moge_pointcloud", rr.Points3D(self.point_list[j], colors=self.color_list[j]))
            # rr.log(f"view-{j}/subsampled_pointcloud", rr.Points3D(self.pcls[j].points_list(),
            #                                                       colors=self.pcls[j].features_list()))

            rr.log(f"view-{j}/mhr_model", rr.Mesh3D(vertex_positions=meshes[j].detach().cpu().numpy(),
                                                    triangle_indices=self.faces.cpu(),
                                                    # vertex_normals=meshes[j].vertex_normals,
                                                    vertex_colors=np.array([0, 0, 190])))

            image = np.zeros([self.mask_list[0].shape[0], self.mask_list[0].shape[1], 3])
            image[:, :, 0] = self.mask_list[j]
            image[:, :, 1] = rendered_masks[j].detach().cpu().numpy()

            rr.log(f"view-{j}/mask_diff", rr.Image(image))



if __name__ == '__main__':
    image_dir = 'D:/Research/data/antropo/x1'
    data_dir = 'D:/Research/data/antropo/x1'

    fnames = ['IMG_9576.jpeg', 'IMG_9579.jpeg', 'IMG_9581.jpeg', 'IMG_9584.jpeg']
    # fnames = ['IMG_9581.jpeg']

    model = MultiViewMultiPoseMHR(image_dir, data_dir, fnames)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    pbar = tqdm(range(500))
    for i in pbar:
        optimizer.zero_grad()
        vertices, masks, loss_pcl, loss_mask = model()
        loss = loss_mask + loss_pcl
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Total: {loss.item():0.6f}, Mask: {loss_mask.item():0.6f}, PCL: {loss_pcl.item():0.6}")
        if i % 10 == 0:
            model.visualize(i, vertices, masks)







