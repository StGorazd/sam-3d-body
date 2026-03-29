import cv2
import json
import numpy as np
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.loss import point_mesh_face_distance
from pytorch3d.ops import knn_points
from pytorch3d.renderer import FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, PointLights, TexturesVertex, look_at_view_transform

import scipy
import torch
import rerun as rr
import trimesh
from timm.models import model_parameters
from tqdm import tqdm
import time
from mhr.mhr import MHR
from pathlib import Path

from scipy.spatial import cKDTree

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
            lod=3
        )
        faces = torch.as_tensor(
            self.mhr_model.character.mesh.faces,
            dtype=torch.int64,
            device="cuda"
        )
        # print(self.mhr_model.character.mesh.faces)
        # print(type(self.mhr_model.character.mesh.faces[0][0]))
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

        self.register_buffer("target_points", self.pcl.points_padded())
        self.colors = pcl_mesh.colors

        print("Data loaded")

        self.identity = torch.nn.Parameter(torch.zeros([1, 45]), requires_grad=True)
        # self.expr = torch.nn.Parameter(torch.zeros([1, 72]), requires_grad=True)
        # Excluding face expression from optimizing
        self.register_buffer("expr", torch.zeros([1, 72], device="cuda"))
        self.pose = torch.nn.Parameter(torch.zeros([1, 204]), requires_grad=True)
        self.t = torch.nn.Parameter(torch.zeros([1, 3]), requires_grad=True)

        self.register_buffer("finger_pose_mask", self._create_finger_mask())
        self.register_buffer("foot_pose_mask", self._create_foot_mask())

        # Setting initial rotation to wrist to help fitting
        wrists = self._get_wrist_pose_indices()
        with torch.no_grad():
            self.pose[0, wrists['r_uparm_twist']] = -1
            self.pose[0, wrists['l_uparm_twist']] = -1

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

    def _create_foot_mask(self):
        num_total_params = self.pose.shape[1]

        num_pose_param = int(self.mhr_model.character.parameter_transform.pose_parameters.sum() - 6)
        num_scale_params = int(self.mhr_model.character.parameter_transform.scaling_parameters.sum())

        lbs_parameter_names = self.mhr_model.character.parameter_transform.names[
            6: 6 + num_pose_param + num_scale_params
        ]

        # foot_parts = {
        # "r_foot_bend"
        # "r_foot_lean0"
        # "r_foot_lean1"
        # "r_ball_bend"
        # "l_foot_bend"
        # "l_foot_lean0"
        # "l_foot_lean1"
        # "l_ball_bend"
        # "l_foot_ry_flexible"
        # "l_ball_rx_flexible"
        # "r_foot_ry_flexible"
        # "r_ball_rx_flexible"
        # }
        foot_parts = {"foot", "ball"}

        mask = torch.ones(num_total_params, dtype=torch.float32, device="cuda")

        for i, name in enumerate(lbs_parameter_names[:num_pose_param]):
            if any(part in name.lower() for part in foot_parts):
                mask[6 + i] = 0.0

        return mask.unsqueeze(0)  # shape [1, 204]

    def _get_wrist_pose_indices(self):
        num_pose_param = int(self.mhr_model.character.parameter_transform.pose_parameters.sum() - 6)
        num_scale_params = int(self.mhr_model.character.parameter_transform.scaling_parameters.sum())

        lbs_parameter_names = self.mhr_model.character.parameter_transform.names[
            6: 6 + num_pose_param + num_scale_params
        ]

        wrist_parts = {"uparm"}

        wrist_dict = {}

        for i, name in enumerate(lbs_parameter_names[:num_pose_param]):
            lname = name.lower()
            if any(part in lname for part in wrist_parts):
                wrist_dict[name] = 6 + i

        return wrist_dict

    def forward(self, mesh_samples: int = 5_000, single_direction: bool = False):
        t1 = self._start_timer()
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

        # t4 = self._start_timer()
        # pcl_loss = pytorch3d.loss.point_mesh_face_distance(meshes, self.pcl)
        # ======================================================================
        sampled_points = sample_points_from_meshes(meshes, num_samples=mesh_samples)
        pcl_loss, _ = pytorch3d.loss.chamfer_distance(self.target_points, sampled_points, single_directional=single_direction)
        # loss_time = self._end_timer(t4)

        # total_time = mhr_time + transform_time + mesh_creation_time + loss_time
        # print(f"[ms] MHR={mhr_time:.2f} T={transform_time:.2f} M={mesh_creation_time:.2f} L={loss_time:.2f} Total={total_time:.2f}")

        return mean_model_vertices, pcl_loss

    def _compute_vertex_error_colors(self, vertices):
        knn = knn_points(vertices, self.target_points, K=1)
        vertex_errors = torch.sqrt(knn.dists[0, :, 0].clamp_min(1e-12))  # meters

        norm = (vertex_errors / 0.01).clamp(0.0, 1.0)

        r = (255.0 * norm).to(torch.uint8)
        g = torch.zeros_like(r)
        b = (255.0 * (1.0 - norm)).to(torch.uint8)

        vertex_colors = torch.stack([r, g, b], dim=1)
        return vertex_colors, vertex_errors

    def visualize(self, i, meshes):
        rr.set_time("frame_idx", sequence=i)

        rr.log("pointcloud", rr.Points3D(self.pcl[0].cpu().points_packed().numpy(), colors=self.colors))

        vertex_colors, vertex_errors = self._compute_vertex_error_colors(meshes)

        rr.log("mhr_model", rr.Mesh3D(vertex_positions=meshes[0].detach().cpu().numpy(),
                                                 triangle_indices=self.faces[0].cpu(),
                                                 # vertex_colors=np.array([0, 0, 190])))
                                                 vertex_colors=vertex_colors.cpu().numpy()
                                      )
               )

    def save_parameters(self, path: str):
        params = {
            "identity": self.identity.detach().cpu(),
            "pose": self.pose.detach().cpu(),
            "t": self.t.detach().cpu(),
            "expr": self.expr.detach().cpu(),
        }
        torch.save(params, path)

    def save_obj(self, path: str, vertices: torch.Tensor):
        # vertices: [1, V, 3] or [V, 3]
        if vertices.dim() == 3:
            vertices = vertices[0]

        v = vertices.detach().cpu().numpy()
        f = self.faces[0].detach().cpu().numpy()

        mesh = trimesh.Trimesh(vertices=v, faces=f)
        mesh.export(path)

    def save_front_render(self, path: str, vertices: torch.Tensor, image_width: int = 768, image_height: int = 1300):
        device = vertices.device
        faces = self.faces.to(device)

        # Center and size of fitted mesh
        verts = vertices[0]
        vmin = verts.min(dim=0).values
        vmax = verts.max(dim=0).values
        center = (vmin + vmax) / 2.0
        extent = (vmax - vmin).max().item()

        # White mesh
        verts_rgb = torch.ones_like(vertices, device=device) * 0.85
        textures = TexturesVertex(verts_features=verts_rgb)

        mesh = Meshes(verts=vertices, faces=faces, textures=textures)

        # Front camera:
        dist = max(extent * 2.25, 1.5)
        eye = center + torch.tensor([0.0, 0.0, -dist], device=device)
        at = center.unsqueeze(0)
        up = torch.tensor([[0.0, -1.0, 0.0]], device=device)

        R, T = look_at_view_transform(eye=eye.unsqueeze(0), at=at, up=up)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=20.0)

        raster_settings = RasterizationSettings(
            image_size=(image_height, image_width),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        lights = PointLights(
            device=device,
            location=eye.unsqueeze(0)
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings,
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights,
            ),
        )

        images = renderer(mesh)
        image = images[0, ..., :3].detach().cpu().numpy()
        image = (image * 255).clip(0, 255).astype(np.uint8)

        # RGB -> BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)

    def save_final_rrd(self, path: str, vertices: torch.Tensor, frame_idx: int = 0):
        rec = rr.RecordingStream("final_fit")
        rec.save(path)

        rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rec.log(
            "world/XYZ",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
            static=True,
        )

        rec.set_time("frame_idx", sequence=frame_idx)

        rec.log(
            "pointcloud",
            rr.Points3D(
                self.pcl[0].cpu().points_packed().numpy(),
                colors=self.colors,
            ),
        )

        vertex_colors, _ = self._compute_vertex_error_colors(vertices)

        rec.log(
            "mhr_model",
            rr.Mesh3D(
                vertex_positions=vertices[0].detach().cpu().numpy(),
                triangle_indices=self.faces[0].cpu().numpy(),
                vertex_colors=vertex_colors.cpu().numpy(),
            ),
        )

    def _distance_stats(self, dists: torch.Tensor, prefix: str):
        mean_dist = dists.mean()
        median_dist = dists.median()
        std_dist = dists.std(unbiased=False)
        max_dist = dists.max()
        rmse = torch.sqrt((dists ** 2).mean())

        p90 = torch.quantile(dists, 0.90)
        p95 = torch.quantile(dists, 0.95)
        p99 = torch.quantile(dists, 0.99)

        inlier_2mm = (dists < 0.002).float().mean()
        inlier_5mm = (dists < 0.005).float().mean()
        inlier_10mm = (dists < 0.010).float().mean()
        inlier_20mm = (dists < 0.020).float().mean()
        inlier_50mm = (dists < 0.050).float().mean()

        return {
            f"{prefix}_mean": float(mean_dist.item()),
            f"{prefix}_median": float(median_dist.item()),
            f"{prefix}_std": float(std_dist.item()),
            f"{prefix}_max": float(max_dist.item()),
            f"{prefix}_p90": float(p90.item()),
            f"{prefix}_p95": float(p95.item()),
            f"{prefix}_p99": float(p99.item()),
            f"{prefix}_rmse": float(rmse.item()),
            f"{prefix}_inlier_ratio_2mm": float(inlier_2mm.item()),
            f"{prefix}_inlier_ratio_5mm": float(inlier_5mm.item()),
            f"{prefix}_inlier_ratio_10mm": float(inlier_10mm.item()),
            f"{prefix}_inlier_ratio_20mm": float(inlier_20mm.item()),
            f"{prefix}_inlier_ratio_50mm": float(inlier_50mm.item()),
        }

    def compute_fit_metrics(self, per_point_path: str, vertices: torch.Tensor, num_samples: int = 100_000):
        device = vertices.device
        faces = self.faces.to(device)
        mesh = Meshes(verts=vertices, faces=faces)

        sampled_points = sample_points_from_meshes(mesh, num_samples=num_samples)
        target_points = self.target_points

        # point cloud -> mesh samples
        knn_p2m = knn_points(target_points, sampled_points, K=1)
        dists_p2m = torch.sqrt(knn_p2m.dists[0, :, 0].clamp_min(1e-12))

        np.save(per_point_path, dists_p2m.detach().cpu().numpy())

        # mesh samples -> point cloud
        knn_m2p = knn_points(sampled_points, target_points, K=1)
        dists_m2p = torch.sqrt(knn_m2p.dists[0, :, 0].clamp_min(1e-12))

        metrics = {}
        metrics.update(self._distance_stats(dists_p2m, "point_to_mesh"))
        metrics.update(self._distance_stats(dists_m2p, "mesh_to_point"))

        metrics["number_of_points"] = int(target_points.shape[1])
        return metrics

    def save_json_data(self, path: str,
                       per_point_path: str,
                       vertices,
                       loss_value: float,
        ):
        mesh_samples = 100_000

        data = {
            "loss": float(loss_value),
            "mesh_samples": mesh_samples,
            **self.compute_fit_metrics(per_point_path, vertices, mesh_samples)
        }

        with open(path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4)




if __name__ == '__main__':
    # image_dir = 'D:/Research/data/antropo/x1'
    # data_dir = 'D:/Research/data/antropo/x1'

    # fname = 'D:/Research/data/antropo/pointcloud/body_processed.ply'
    # fname = "pointclouds/body_processed.ply"
    # fnames = ['IMG_9581.jpeg']

    root = Path("/home/stg/Dev/research_batch_2026")
    out_root = Path("./out")

    last_loss = 0.0

    to_process = 1
    pbar = tqdm(root.iterdir(), desc="Processing avatars", total=to_process)
    for src_data in pbar:
        print("===========================" , to_process , "to process", "===========================")
        if not src_data.is_dir() or src_data.name.startswith("__"):
            continue

        target_dir = out_root / src_data.name
        target_dir.mkdir(parents=False, exist_ok=False)

        point_cloud_path = src_data / "body_processed.ply"

        model = SinglePoseMHR(point_cloud_path)
        model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.032)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',
        #     factor=0.75,
        #     patience=25,
        #     min_lr=1e-5
        # )

        # pbar = tqdm(range(300))
        # for i in pbar:
        for i in range(300):
            if i < 75:
                samples = 3_000
            elif i < 150:
                samples = 6_000
            elif i < 220:
                samples = 10_000
            else:
                samples = 40_000

            if i < 100:
                single_direction = True
            else:
                single_direction = False

            if i == 100:
                for group in optimizer.param_groups:
                    group['lr'] = 0.025
            elif i == 150:
                for group in optimizer.param_groups:
                    group['lr'] = 0.020
            elif i == 250:
                for group in optimizer.param_groups:
                    group['lr'] = 0.01


            optimizer.zero_grad()
            vertices, loss = model(mesh_samples=samples, single_direction=single_direction)
            loss.backward()

            if i < 150:
                with torch.no_grad():
                    # model.pose.grad *= model.foot_pose_mask
                    model.pose.grad *= model.finger_pose_mask


            optimizer.step()

            # scheduler.step(loss.item())
            currentLr = optimizer.param_groups[0]['lr']

            pbar.set_description(f"Loss: {loss.item():0.6f}, LR: {currentLr:0.6f}")
            if i % 10 == 0:
                model.visualize(i, vertices)

        final_loss = loss.item()
        model.save_parameters(str(target_dir / "mhr_params.pt"))
        model.save_obj(str(target_dir / "mhr_model.obj"), vertices)
        model.save_front_render(str(target_dir / "front.png"), vertices)
        model.save_final_rrd(str(target_dir / "final_fit.rrd"), vertices)
        model.save_json_data(str(target_dir / "data.json"),
                             str(target_dir / "point_to_mesh_distances.npy"),
                             vertices,
                             last_loss,
        )


        to_process -= 1
        if not to_process:
            break