"""Fit a single MHR body to several views/poses of the same person at once.

For every image in --fnames the three files prepared by extract_mhr_depth.py are read from
data_dir (<name>_mask.jpg, <name>_depth.npy, <name>.json) and used to optimize:

    identity, expression   - shared by all views, this is the body we are after
    pose, translation      - per view, the person moves between the shots

against three terms: a soft silhouette rendered with pytorch3d versus the masks, and the
point->surface and front-facing-surface->point distances versus the MoGe point clouds.
Progress is streamed to a rerun viewer that is spawned on startup.

Every run writes <out_dir>/[<subdir>/]<name>.json with the fitted parameters and <name>.obj with
the neutral mesh of that fit, its vertices coloured by the mean of the point clouds that saw
them. <name> is `model` for a plain run and `model_<combination>` with --all.

Example:
    python joint_optim.py
    python joint_optim.py --image_dir <dir> --data_dir <dir> --fnames a.jpg b.jpg
    python joint_optim.py --image_dir <dir> --data_dir <dir> --all_images --max_iters 800
    python joint_optim.py --image_dir <dir> --data_dir <dir> --subdirs --all

All the views of one run have to be the same person, so --all_images (every image directly
inside the fitted folder) deliberately does not recurse the way extract_mhr_depth.py --subdirs
does. --subdirs instead treats every subdirectory of image_dir as its own subject and fits them
one after another, mirroring the relative structure inside out_dir. Without --subdirs the
relative names work as well (--fnames sub/a.jpg sub/b.jpg).

--all repeats the fit for several combinations of the views (all of them, front+back, front
only) and writes one json per combination, so that a full fit can be compared against the two
and one view ones.

The MoGe clouds are rescaled with utils.pointcloud.get_scaled_pointcloud on the 8x5 checkerboard
that should be visible in every image; a view without one falls back to the metric scale MoGe-2
predicted for it.
"""

import argparse
import json
import os
import traceback
import warnings

import cv2
import numpy as np
import pytorch3d
import pytorch3d
from pytorch3d.renderer import RasterizationSettings, MeshRenderer, MeshRasterizer, SoftSilhouetteShader, BlendParams, \
    FoVPerspectiveCameras
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.loss.point_mesh_distance import point_face_distance, face_point_distance

import scipy
from scipy.spatial import cKDTree
import torch
import rerun as rr
import trimesh
from tqdm import tqdm

from utils.image import load_image
from utils.measure import get_measurements
from utils.paths import depth_path, is_prepared, json_path, list_images, mask_path, stem
from utils.pointcloud import get_moge_pointcloud, get_scaled_pointcloud

DEFAULT_DIR = 'D:/Research/data/antropo/mini_scanovaci_den'
DEFAULT_FNAMES = ['front.JPG', 'back.JPG', 'left.JPG', 'right.JPG']

# the three folders of a session, relative to --dir
IMAGE_SUBDIR = 'foto'
DATA_SUBDIR = 'processed'
OUT_SUBDIR = 'output'


def scaled_pointcloud(image, moge_depth, K, fname=''):
    """Metric MoGe cloud, rescaled on the checkerboard when there is one in the image.

    extract_mhr_depth.py stores the raw MoGe-2 depth, which is already metric, so an image
    without a visible board is not lost -- it just keeps the (less accurate) scale MoGe
    predicted for it instead of the one measured on the 8x5 board.
    """
    try:
        return get_scaled_pointcloud(image, moge_depth, K)
    except ValueError:
        print(f"{fname}: no 8x5 checkerboard found, falling back to the MoGe-2 metric scale")
        return get_moge_pointcloud(moge_depth, K)


class MultiViewMultiPoseMHR(torch.nn.Module):
    def __init__(self, image_dir, data_dir, fnames, subsample=5000, mask_scale_factor=0.125,
                 min_triangle_area=1e-12, max_faces_per_bin=None, device='cuda', run_name='Viewer',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the point clouds and the cameras are not plain buffers, they are built on this device
        # directly, so a later .to() of the module has to use the same one
        self.device = torch.device(device)
        self.mhr_model = torch.jit.load("assets/mhr_model.pt")
        self.register_buffer('faces', self.mhr_model.character_torch.mesh.faces)
        self.n_views = len(fnames)
        self.subsample = subsample
        self.mask_scale_factor = mask_scale_factor
        self.min_triangle_area = min_triangle_area

        if max_faces_per_bin is None:
            max_faces_per_bin = max(10000, int(self.faces.shape[0]) // 2)
        self.max_faces_per_bin = max_faces_per_bin

        # One rerun recording per run, so that a sweep does not overwrite the views of the
        # previous fit. spawn=True reuses the viewer that is already listening.
        rr.init(run_name, spawn=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log("world/XYZ", rr.Arrows3D(vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))

        pose_param_list = []
        identity_params_list = []
        expr_params_list = []
        t_list = []
        self.fov_list = []
        self.focal_list = []

        self.point_list = []
        self.color_list = []
        self.mask_list = []

        print("Loading detections")
        for fname in fnames:
            print(fname)
            image = load_image(os.path.join(image_dir, fname))
            mask = load_image(mask_path(data_dir, fname))[:, :, 0] / 255
            mask_pcl = cv2.erode(mask, np.ones((10, 10), np.uint8), iterations=1) > 0.5

            moge_depth = np.load(depth_path(data_dir, fname))

            with open(json_path(data_dir, fname), 'r') as f:
                json_data = json.load(f)

            f = json_data['focal_length']
            # PyTorch3D maps NDC [-1, 1] onto the smaller image dimension, and
            # FoVPerspectiveCameras(aspect_ratio=1) maps that range to a half-angle of fov / 2,
            # so fov has to be measured across the smaller side. Hardcoding shape[1] happens to be
            # correct for portrait photos only and breaks silently on landscape ones.
            fov = 2 * np.rad2deg(np.arctan(min(mask.shape[0], mask.shape[1]) / (2 * f)))
            self.fov_list.append(fov)
            self.focal_list.append(float(f))
            K = np.array([[f, 0, image.shape[1] / 2], [0, f, image.shape[0] / 2], [0, 0, 1]])
            points = scaled_pointcloud(image, moge_depth, K, fname)

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

        # The input clouds never change during optimization, so log them once as static data
        # rather than re-streaming every point over gRPC from visualize() on each logged step.
        for j in range(self.n_views):
            rr.log(f"view-{j}/moge_pointcloud",
                   rr.Points3D(self.point_list[j], colors=self.color_list[j]), static=True)

        self.identity = torch.nn.Parameter(torch.from_numpy(np.mean(np.array(identity_params_list), axis=0).astype(np.float32)), requires_grad=True)
        self.expr = torch.nn.Parameter(torch.from_numpy(np.mean(np.array(expr_params_list), axis=0).astype(np.float32)), requires_grad=True)
        self.poses = torch.nn.Parameter(torch.from_numpy(np.array(pose_param_list, dtype=np.float32)), requires_grad=True)
        self.ts = torch.nn.Parameter(torch.from_numpy(np.array(t_list, dtype=np.float32)), requires_grad=True)

        self.pcls = Pointclouds([torch.from_numpy(x.astype(np.float32)) for x in self.point_list],
                                [torch.from_numpy(x.astype(np.float32)) for x in self.color_list]).subsample(self.subsample).to(self.device)
        # Cached so the loss does not trigger a device sync (.item()) on every iteration.
        self.max_points_per_cloud = int(self.pcls.num_points_per_cloud().max().item())
        self.max_faces_per_mesh = int(self.faces.shape[0])

        for i in range(self.n_views):
            self.mask_list[i] = cv2.resize(self.mask_list[i], None,
                                           fx=self.mask_scale_factor, fy=self.mask_scale_factor,
                                           interpolation=cv2.INTER_AREA)
        self.register_buffer('masks', torch.from_numpy(np.array(self.mask_list).astype(np.float32)))

        self.cameras = FoVPerspectiveCameras(fov=self.fov_list, device=self.device)
        blend_params = BlendParams(sigma=1e-5, gamma=1e-6)
        raster_settings = RasterizationSettings(
            image_size=(self.mask_list[0].shape[0], self.mask_list[0].shape[1]),
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=64,
            # bin_size=0 forces naive rasterization, which tests every face against every pixel.
            # None selects the binned coarse-to-fine path: ~1.6x faster with identical gradients.
            # faces_per_pixel must stay high -- the silhouette gradient lives in the thin blurred
            # boundary band, and lowering it to 16 drops gradient cosine to 0.56.
            bin_size=None,
            max_faces_per_bin=self.max_faces_per_bin,
        )

        # buffers, so that .to() moves them along with the masks and the faces
        self.register_buffer('R', torch.diag(torch.tensor([-1, -1, 1], dtype=torch.float32))
                             .unsqueeze(0).expand(self.n_views, -1, -1).contiguous())
        self.register_buffer('T', torch.zeros(self.n_views, 3))

        # Create a silhouette mesh renderer by composing a rasterizer and a shader.
        self.silhouette_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=self.cameras,
                                                                          raster_settings=raster_settings),
                                                shader=SoftSilhouetteShader(blend_params=blend_params))

    def point_to_surface_loss(self, meshes):
        """Mean squared distance from each observed point to the closest mesh face.

        Only the point->face direction is used. point_mesh_face_distance() also adds the
        face->point term, but the clouds are single-view partial scans, so that term scores the
        occluded back of the body as ~9 cm of error and drags the mesh toward the visible surface
        (it measured 7355x larger than the point->face term).

        min_triangle_area must sit well below the mesh face areas (mean 4.7e-05 m^2 here); the
        pytorch3d default of 5e-3 classifies 100% of the faces as degenerate and inflates the
        distance by ~4000x.
        """
        return point_face_distance(
            self.pcls.points_packed(),
            self.pcls.cloud_to_packed_first_idx(),
            meshes.verts_packed()[meshes.faces_packed()],
            meshes.mesh_to_faces_packed_first_idx(),
            self.max_points_per_cloud,
            self.min_triangle_area,
        ).mean()  # clouds are equal-sized after subsample(), so mean == per-cloud weighting

    def visible_surface_loss(self, meshes):
        """Mean squared distance from each *front-facing* mesh face to the closest observed point.

        This is the term that pulls the mesh onto the cloud. point_to_surface_loss() alone cannot
        do it: the clouds already sit ~0.2 mm from the surface for every candidate mesh (including
        meshes fitted to other views), so it is saturated and carries no usable gradient.

        The full face->point term is unusable here because the clouds are single-view partial
        scans -- back-facing geometry has no support and measures ~8-12 cm, versus ~1-8 cm for
        front-facing geometry. Restricting the term to front-facing faces keeps the real signal
        and discards the unobservable part. The camera sits at the origin in these coordinates,
        so a face is front-facing when its normal points back toward it.
        """
        tris = meshes.verts_packed()[meshes.faces_packed()]
        with torch.no_grad():
            normals = torch.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0], dim=-1)
            front = ((normals * tris.mean(1)).sum(-1) < 0).float()
        face_dists = face_point_distance(
            self.pcls.points_packed(),
            self.pcls.cloud_to_packed_first_idx(),
            tris,
            meshes.mesh_to_faces_packed_first_idx(),
            self.max_faces_per_mesh,
            self.min_triangle_area,
        )
        return (face_dists * front).sum() / front.sum().clamp(min=1.0)

    def posed_vertices(self):
        """The shared body under the per-view poses, in the camera frame of each view."""
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
        return mean_model_vertices

    def forward(self):
        mean_model_vertices = self.posed_vertices()

        meshes = Meshes(mean_model_vertices, self.faces.unsqueeze(0).expand(self.n_views, -1, -1))
        pcl_loss = self.point_to_surface_loss(meshes)
        vis_loss = self.visible_surface_loss(meshes)

        rendered_masks = self.silhouette_renderer(meshes, R=self.R, T=self.T)[..., 3]
        mask_loss = torch.mean((rendered_masks - self.masks) ** 2)

        return mean_model_vertices, rendered_masks, pcl_loss, vis_loss, mask_loss

    def visualize(self, i, meshes, rendered_masks):
        rr.set_time("frame_idx", sequence=i)

        for j in range(self.n_views):
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



class ConvergenceMonitor:
    def __init__(self, window=100, tol=0.01, patience=2, min_iters=300):
        self.window = window
        self.tol = tol
        self.patience = patience
        self.min_iters = min_iters
        self.losses = []
        self.prev_mean = None
        self.strikes = 0
        self.reason = None

    def converged(self, loss):
        self.losses.append(loss)
        n = len(self.losses)
        if n < self.min_iters or n % self.window:
            return False
        current = sum(self.losses[-self.window:]) / self.window
        if self.prev_mean is not None:
            rel = (self.prev_mean - current) / self.prev_mean
            self.strikes = self.strikes + 1 if rel < self.tol else 0
            if self.strikes >= self.patience:
                self.reason = (f"converged after {n} iterations: mean loss over the last "
                               f"{self.window} improved by less than {self.tol:.1%} for "
                               f"{self.patience} consecutive windows (last {rel:+.2%})")
                return True
        self.prev_mean = current
        return False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dir', default=DEFAULT_DIR,
                        help=f'top level folder of the session, holding {IMAGE_SUBDIR}/, '
                             f'{DATA_SUBDIR}/ and {OUT_SUBDIR}/')
    parser.add_argument('--image_dir', default=None, help=f'input images (default: <dir>/{IMAGE_SUBDIR})')
    parser.add_argument('--data_dir', default=None,
                        help=f'files prepared by extract_mhr_depth.py (default: <dir>/{DATA_SUBDIR})')
    parser.add_argument('--out_dir', default=None,
                        help=f'fitted models (default: <dir>/{OUT_SUBDIR})')
    parser.add_argument('--fnames', nargs='*', default=DEFAULT_FNAMES, help='image file names inside image_dir')
    parser.add_argument('--all_images', action='store_true',
                        help='use every image of the fitted folder instead of --fnames')
    parser.add_argument('--all', action='store_true',
                        help='fit several view combinations one after another (all views, '
                             'front+back, front only), one json per combination')
    parser.add_argument('--subdirs', action='store_true',
                        help='fit every subdirectory of image_dir as its own subject, '
                             'the structure of image_dir is mirrored inside out_dir')
    parser.add_argument('--overwrite', action='store_true', help='redo runs whose json already exists')

    parser.add_argument('--subsample', type=int, default=5000, help='points kept per view point cloud')
    parser.add_argument('--mask_scale_factor', type=float, default=0.125,
                        help='downscale applied to the masks, sets the silhouette render resolution')
    parser.add_argument('--min_triangle_area', type=float, default=1e-12,
                        help='faces below this area are treated as degenerate by the distance terms')
    parser.add_argument('--max_faces_per_bin', type=int, default=None,
                        help='rasterizer bin capacity (default: half the face count)')

    parser.add_argument('--w_pcl', type=float, default=1e4, help='weight of the point->surface term')
    parser.add_argument('--w_vis', type=float, default=30.0, help='weight of the visible surface->point term')

    parser.add_argument('--max_iters', type=int, default=400, help='length of the annealed schedule')
    parser.add_argument('--lr_ts', type=float, default=3e-2, help='lr of the per-view translations')
    parser.add_argument('--lr_poses', type=float, default=1e-2, help='lr of the per-view poses')
    parser.add_argument('--lr_identity', type=float, default=3e-2, help='lr of the shared identity coeffs')
    parser.add_argument('--lr_expr', type=float, default=1e-3, help='lr of the shared expression coeffs')
    parser.add_argument('--eta_min', type=float, default=1e-4, help='final lr of the cosine annealing schedule')

    parser.add_argument('--conv_window', type=int, default=50, help='iterations averaged by the convergence monitor')
    parser.add_argument('--conv_tol', type=float, default=0.005,
                        help='relative improvement per window that still counts as progress')
    parser.add_argument('--conv_patience', type=int, default=2, help='stalled windows before the run is stopped')
    parser.add_argument('--conv_min_iters', type=int, default=200, help='iterations before convergence is checked')
    parser.add_argument('--vis_every', type=int, default=10, help='log to rerun every N iterations')
    parser.add_argument('--device', default='cuda')

    return parser.parse_args()


def resolve_dirs(args):
    """Fill in the three folders of the session from --dir, unless they were given explicitly."""
    args.image_dir = args.image_dir or os.path.join(args.dir, IMAGE_SUBDIR)
    args.data_dir = args.data_dir or os.path.join(args.dir, DATA_SUBDIR)
    args.out_dir = args.out_dir or os.path.join(args.dir, OUT_SUBDIR)
    return args


def subject_dirs(image_dir):
    """Relative paths of the subdirectories of image_dir that directly contain images."""
    dirs = sorted({os.path.dirname(x) for x in list_images(image_dir, recursive=True)})
    subdirs = [x for x in dirs if x]
    if '' in dirs and subdirs:
        print("--subdirs: images directly inside image_dir are ignored, only subdirectories are fitted")
    return subdirs or ['']


def collect_fnames(args, subdir):
    """Usable views of one subject, as names relative to image_dir."""
    if args.all_images:
        names = list_images(os.path.join(args.image_dir, subdir))
    else:
        names = args.fnames
    names = [f'{subdir}/{x}' if subdir else x for x in names]

    usable = [x for x in names if os.path.exists(os.path.join(args.image_dir, x))
              and is_prepared(args.data_dir, x)]
    for missing in [x for x in names if x not in usable]:
        print(f"{missing}: missing image or not prepared by extract_mhr_depth.py, skipping")
    return usable


def output_subdir(args, fnames):
    """The folder inside out_dir that the results of these views belong to.

    It is the folder the views themselves live in: with --subdirs (or --fnames sub/a.jpg) that
    is the subdirectory, mirroring image_dir inside out_dir. When the views sit directly in
    image_dir, image_dir is itself the folder of one subject, so its own name is used -- without
    this a run of foto/viktor_calib and a run of foto/zuzka_calib would both write output/model.json.
    """
    dirs = {os.path.dirname(x) for x in fnames}
    subdir = dirs.pop() if len(dirs) == 1 else ''
    if subdir:
        return subdir

    name = os.path.basename(os.path.abspath(args.image_dir))
    return '' if name == IMAGE_SUBDIR else name  # <dir>/foto holds the subjects, it is not one


def view_combos(fnames, use_all):
    """[(output name, views)] - a single run, or the combinations requested by --all."""
    if not use_all:
        return [('model', fnames)]

    def pick(*wanted):
        return [x for x in fnames if os.path.basename(stem(x)).lower() in wanted]

    combos = [('model_all', list(fnames)), ('model_front_back', pick('front', 'back')),
              ('model_front', pick('front'))]

    # front+back collapses onto model_all when those are the only views, keep the first name
    seen, out = set(), []
    for name, views in combos:
        if views and tuple(views) not in seen:
            seen.add(tuple(views))
            out.append((name, views))
    return out


def save_result(model, fnames, losses, out_path):
    """Write the fitted parameters: identity and expression are shared, pose and t are per view."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    poses = model.poses.detach().cpu().numpy()
    ts = model.ts.detach().cpu().numpy()

    data = {
        'fnames': list(fnames),
        'shape_params': model.identity.detach().cpu().numpy().tolist(),
        'expr_params': model.expr.detach().cpu().numpy().tolist(),
        'views': [{'fname': fname,
                   'focal_length': model.focal_list[i],
                   'mhr_model_params': poses[i].tolist(),
                   'pred_cam_t': ts[i].tolist()} for i, fname in enumerate(fnames)],
        'losses': losses,
    }
    with open(out_path, 'w') as f:
        json.dump(data, f)
    print(f"saved {out_path}")


def neutral_vertices(model):
    """The fitted body in the rest pose, in metres and in the MHR frame.

    Same neutral parameters as utils.measure.get_base_mesh_and_skeleton: translation, global
    rotation and the lbs parameters all zero. The fitted expression is kept, it is part of the
    body we solved for. No y/z flip and no per-view translation, so the mesh is y-up around
    the origin instead of sitting in one of the camera frames.
    """
    with torch.no_grad():
        rest_pose = torch.zeros_like(model.poses[:1])
        vertices, skel_state = model.mhr_model(model_parameters=rest_pose,
                                               identity_coeffs=model.identity.unsqueeze(0),
                                               face_expr_coeffs=model.expr.unsqueeze(0))
    return vertices[0].cpu().numpy() / 100.0


def mean_cloud_colors(model, posed_vertices, max_dist=0.02, unseen=(153, 153, 153)):
    """Per vertex mean over the views of the colour of the closest point of that view's cloud.

    The clouds are single-view scans, so a vertex is only sampled from the views that actually
    observed it: the camera sits at the origin of the posed frames, so a vertex has to face it
    and to have a cloud point within max_dist. Vertices no view saw stay grey. The mesh topology
    is shared, so colours gathered in the posed frames apply to the neutral mesh unchanged.
    """
    faces = model.faces.cpu().numpy()
    total = np.zeros((posed_vertices.shape[1], 3))
    count = np.zeros(posed_vertices.shape[1])

    for j in range(model.n_views):
        vertices = posed_vertices[j]
        normals = trimesh.Trimesh(vertices, faces, process=False).vertex_normals
        front = (normals * vertices).sum(-1) < 0
        dist, idx = cKDTree(model.point_list[j]).query(vertices)
        take = front & (dist < max_dist)
        total[take] += model.color_list[j][idx[take]]
        count[take] += 1

    seen = count > 0
    colors = np.tile(np.array(unseen, dtype=np.float64), (len(count), 1))
    colors[seen] = total[seen] / count[seen, np.newaxis]
    return colors, seen


def save_obj(vertices, faces, colors, out_path):
    """OBJ with per-vertex colours as the three extra floats of the v lines."""
    with open(out_path, 'w') as f:
        np.savetxt(f, np.hstack([vertices, colors / 255]), fmt='v %.6f %.6f %.6f %.4f %.4f %.4f')
        np.savetxt(f, faces + 1, fmt='f %d %d %d')  # obj indices are 1 based


def save_mesh(model, out_path):
    """The neutral mesh of the fit, coloured from the clouds of all the views."""
    with torch.no_grad():
        posed = model.posed_vertices().cpu().numpy()
    colors, seen = mean_cloud_colors(model, posed)
    save_obj(neutral_vertices(model), model.faces.cpu().numpy(), colors, out_path)
    print(f"saved {out_path} ({100 * seen.mean():0.0f}% of the vertices coloured from the clouds)")


def optimize(model, args):
    optimizer = torch.optim.Adam([
        {'params': [model.ts],       'lr': args.lr_ts},
        {'params': [model.poses],    'lr': args.lr_poses},
        {'params': [model.identity], 'lr': args.lr_identity},
        {'params': [model.expr],     'lr': args.lr_expr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iters, eta_min=args.eta_min)

    monitor = ConvergenceMonitor(window=args.conv_window, tol=args.conv_tol,
                                 patience=args.conv_patience, min_iters=args.conv_min_iters)

    # the four terms of every iteration, saved with the fit so a run can be plotted afterwards
    trail = {'total': [], 'mask': [], 'pcl': [], 'vis': []}

    pbar = tqdm(range(args.max_iters))
    for i in pbar:
        optimizer.zero_grad()
        vertices, masks, loss_pcl, loss_vis, loss_mask = model()
        loss = loss_mask + args.w_pcl * loss_pcl + args.w_vis * loss_vis
        loss.backward()
        optimizer.step()
        scheduler.step()

        # the terms are stored weighted, the way they enter the total and the progress bar
        trail['total'].append(loss.item())
        trail['mask'].append(loss_mask.item())
        trail['pcl'].append(args.w_pcl * loss_pcl.item())
        trail['vis'].append(args.w_vis * loss_vis.item())

        # waist, zadok = get_measurements(model.mhr_model, model.identity.unsqueeze(0))
        # pbar.set_description(f"Total: {loss.item():0.6f}, Mask: {loss_mask.item():0.6f}, PCL: {loss_pcl.item():0.6}, Waist: {waist:0.4}, Zadok: {zadok:0.4}")
        pbar.set_description(f"Total: {trail['total'][-1]:0.6f}, Mask: {trail['mask'][-1]:0.6f}, "
                             f"PCL: {trail['pcl'][-1]:0.6f}, Vis: {trail['vis'][-1]:0.6f}")
        if i % args.vis_every == 0:
            model.visualize(i, vertices, masks)

        if monitor.converged(trail['total'][-1]):
            model.visualize(i, vertices, masks)
            print()
            print(monitor.reason)
            break
    else:
        print()
        print(f"finished the full {args.max_iters}-iteration annealed schedule")

    losses = {k: v[-1] for k, v in trail.items() if v}
    losses['iters'] = len(trail['total'])
    losses['stop_reason'] = monitor.reason or 'full schedule'
    losses['trail'] = trail
    return losses


def fit(args, fnames, out_path, run_name):
    """One optimization run over one set of views of one subject."""
    print(f"Optimizing over {len(fnames)} views from {args.image_dir} into {out_path}")

    model = MultiViewMultiPoseMHR(args.image_dir, args.data_dir, fnames,
                                  subsample=args.subsample,
                                  mask_scale_factor=args.mask_scale_factor,
                                  min_triangle_area=args.min_triangle_area,
                                  max_faces_per_bin=args.max_faces_per_bin,
                                  device=args.device,
                                  run_name=run_name)
    model.to(args.device)

    losses = optimize(model, args)
    save_result(model, fnames, losses, out_path)
    save_mesh(model, os.path.splitext(out_path)[0] + '.obj')
    return model


def main(args):
    warnings.filterwarnings('error', message='Bin size was too small')

    args = resolve_dirs(args)
    out_dir = args.out_dir
    subdirs = subject_dirs(args.image_dir) if args.subdirs else ['']

    runs = []
    for subdir in subdirs:
        fnames = collect_fnames(args, subdir)
        if not fnames:
            print(f"{subdir or args.image_dir}: no usable views, skipping")
            continue
        for name, views in view_combos(fnames, args.all):
            out_subdir = output_subdir(args, views)
            runs.append((out_subdir, name, views, os.path.join(out_dir, out_subdir, name + '.json')))

    if not args.overwrite:
        todo = [x for x in runs if not os.path.exists(x[3])]
        if len(todo) < len(runs):
            print(f"Skipping {len(runs) - len(todo)} already fitted runs (use --overwrite to redo them)")
        runs = todo

    print(f"{len(runs)} run(s) to fit into {out_dir}")

    model = None
    failed = []
    for subdir, name, views, out_path in runs:
        run_name = f"{subdir} - {name}" if subdir else name
        try:
            model = fit(args, views, out_path, run_name)
        except Exception:
            if len(runs) == 1:
                raise
            # one bad subject should not take the rest of the sweep down with it
            print(f"{run_name}: FAILED")
            traceback.print_exc()
            failed.append(run_name)
        finally:
            if len(runs) > 1:
                # the fits are run one after another, a finished one has no claim on the GPU
                model = None
                torch.cuda.empty_cache()

    if failed:
        print(f"Failed runs: {failed}")

    return model


if __name__ == '__main__':
    main(parse_args())
