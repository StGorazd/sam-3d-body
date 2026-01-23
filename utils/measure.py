import json
import os

import numpy as np
import torch
import trimesh
import rerun as rr
from matplotlib import pyplot as plt


def get_base_mesh_and_skeleton(model, identity_params):
    rot = torch.zeros(1, 3)  # Global Rotation
    trans = torch.zeros(1, 3)  # Translation
    lbs_model_parms = torch.zeros(1, 198)
    params = torch.hstack((trans, rot, lbs_model_parms)).to(identity_params.device)

    identity_coeffs = identity_params
    face_expr_coeffs = torch.zeros(1, 72).to(identity_params.device)

    # Get the mean model mesh.
    mean_model_vertices, skel_state = model(model_parameters=params,
                                            identity_coeffs=identity_coeffs,
                                            face_expr_coeffs=face_expr_coeffs)

    mean_model_vertices = mean_model_vertices.detach().cpu().numpy()[0] / 100.0
    faces = model.character_torch.mesh.faces.cpu().numpy()
    mean_model_mesh = trimesh.Trimesh(mean_model_vertices, faces, process=False)

    skel_state = skel_state.cpu().numpy()[0]
    joint_locations = skel_state[..., :3] / 100.0
    joint_names = model.get_joint_names()

    joint_dict = {k: np.array(v) for k, v in zip(joint_names, joint_locations)}

    return mean_model_mesh, joint_dict

def get_measurements(model, identity_params, visualize=False):
    mesh, joints = get_base_mesh_and_skeleton(model, identity_params)

    spine_start = joints['c_spine0']
    spine_end = joints['c_spine3']

    normal = spine_end - spine_start
    normal = normal / np.linalg.norm(normal)

    xs = np.linspace(0.0, 1.0, 101)
    ys = []

    if visualize:
        rr.init("Measurement visualization", spawn=True)
        rr.log("mhr_model", rr.Mesh3D(vertex_positions=mesh.vertices,
                                      triangle_indices=mesh.faces,
                                      vertex_normals=mesh.vertex_normals,
                                      vertex_colors=np.array([0, 0, 50, 100]),
                                      albedo_factor=np.array([0, 0, 50, 100])))

        # rr.log("slice", rr)
        slice_path = mesh.section(normal, spine_start)
        rr.log("selected_joint", rr.Points3D([spine_start], radii=10))
        line_strips = np.empty((len(slice_path.vertex_nodes), 2, 3))
        line_strips[:, 0, :] = slice_path.vertices[slice_path.vertex_nodes[:, 0]]
        line_strips[:, 1, :] = slice_path.vertices[slice_path.vertex_nodes[:, 1]]
        rr.log("selected_slice", rr.LineStrips3D(line_strips))

    for x in xs:
        point = spine_start + x * (spine_end - spine_start)
        slice_path = mesh.section(normal, point)
        path_2d, _ = slice_path.to_2D(normal=normal)
        lengths = [x.length for x in path_2d.split()]
        ys.append(max(lengths))

    plt.plot(xs, ys)
    plt.show()

    return min(ys), ys[0]


if __name__ == '__main__':
    scripted_mhr_model = torch.jit.load("assets/mhr_model.pt")

    image_name = 'IMG_9581'

    data_dir = 'D:/Research/data/antropo/x1'
    with open(os.path.join(data_dir, f'{image_name}.json'), 'r') as f:
        json_data = json.load(f)

    identity_params = torch.Tensor([json_data['shape_params']])

    get_measurements(scripted_mhr_model, identity_params, visualize=True)

    # mesh, joints = get_base_mesh_and_skeleton(scripted_mhr_model, identity_params)



