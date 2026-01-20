import json
import os

import cv2
import numpy as np
import scipy
import torch
import rerun as rr
import trimesh

from utils.image import load_image
from utils.pointcloud import get_moge_pointcloud, get_scaled_pointcloud


def get_mhr_data(scripted_mhr_model, json_data):
    params = torch.Tensor([json_data['mhr_model_params']])
    params[0, 3:6] = torch.Tensor([json_data['global_rot']])
    params[0, 0:3] = torch.Tensor([json_data['pred_cam_t']])

    identity_coeffs = torch.Tensor([json_data['shape_params']])
    face_expr_coeffs = torch.Tensor([json_data['expr_params']])

    # Get the mean model mesh.
    mean_model_vertices, skel_state = (
        scripted_mhr_model(
            model_parameters=params,
            identity_coeffs=identity_coeffs,
            face_expr_coeffs=face_expr_coeffs,
        )
    )

    mean_model_vertices = mean_model_vertices.numpy()[0] / 100.0
    faces = scripted_mhr_model.character_torch.mesh.faces.cpu().numpy()
    mean_model_mesh = trimesh.Trimesh(mean_model_vertices, faces, process=False)

    # Get the joint locations.
    skel_state = skel_state.numpy()[0]
    joint_locations = skel_state[..., :3] / 100.0
    joint_names = scripted_mhr_model.get_joint_names()

    # Get the skeleton structure.
    joint_parents = scripted_mhr_model.character_torch.skeleton.joint_parents
    joint_parents = np.clip(np.array(joint_parents), 0, np.inf).astype(
        np.int32
    )  # So that the root points to itself.
    parent_joint_locations = joint_locations[joint_parents]

    # Get the kinematic joints (If a joint is a parent of another, then it is a kinematic joint).
    kinematic_joints = joint_locations[np.unique(joint_parents)]
    kinematic_joints_names = [joint_names[i] for i in np.unique(joint_parents)]

    # Get joint local coordinate orientations.
    joint_orientations = skel_state[..., 3:7]
    joint_orientations = scipy.spatial.transform.Rotation.from_quat(joint_orientations).as_matrix()
    kinematic_joint_orientations = joint_orientations[np.unique(joint_parents)]

    return mean_model_mesh


if __name__ == '__main__':
    scripted_mhr_model = torch.jit.load("assets/mhr_model.pt")

    image_dir = 'D:/Research/data/antropo/x1'
    data_dir = 'D:/Research/data/antropo/x1'

    image_name = 'IMG_9576'

    image = load_image(os.path.join(image_dir, f'{image_name}.jpeg'))
    # image = cv2.resize(image, (1536, 2048))
    mask = load_image(os.path.join(data_dir, f'{image_name}_mask.jpg'))[:,:,0] > 177
    moge_depth = np.load(os.path.join(data_dir, f'{image_name}_depth.npy'))

    with open(os.path.join(data_dir, f'{image_name}.json'), 'r') as f:
        json_data = json.load(f)

    print(json_data)

    f = json_data['focal_length']
    K = np.array([[f, 0, image.shape[1]/2], [0, f, image.shape[0]/2], [0, 0, 1]])
    # points = get_moge_pointcloud(moge_depth, K)
    points = get_scaled_pointcloud(image, moge_depth, K)

    masked_points = points.reshape(-1, 3)[mask.ravel()]
    masked_colors = image[:, :, ::-1].reshape(-1, 3)[mask.ravel()]
    masked_points[:, 1] *= -1
    masked_points[:, 2] *= -1

    # write code to visualize the pointcloud using rerun
    rr.init("Viewer", spawn=True)
    rr.log("moge_pointcloud", rr.Points3D(masked_points, colors=masked_colors))

    mean_model_mesh = get_mhr_data(scripted_mhr_model, json_data)

    # mean_model_mesh.vertices[:, 1] *= -1
    # mean_model_mesh.vertices[:, 2] *= -1

    rr.log("mhr_model", rr.Mesh3D(vertex_positions=mean_model_mesh.vertices,
                                  triangle_indices=mean_model_mesh.faces,
                                  vertex_normals=mean_model_mesh.vertex_normals,
                                  # modify this so we get colors that change throughout the mesh
                                  # vertex_colors=np.zeros_like(mean_model_mesh.vertices)))
                                  vertex_colors=np.array([0, 0, 190])))

    # json_mesh = json_data['pred_vertices']
    # rr.log("json_mesh", rr.Points3D(json_mesh, colors=np.array([0, 0, 230])))
