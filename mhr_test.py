from pathlib import Path
import torch
from mhr.mhr import MHR


def get_hand_pose_indices(assets_folder="assets/assets", lod=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mhr_model = MHR.from_files(
        folder=Path(assets_folder),
        device=device,
        lod=lod,
    )

    num_pose_param = int(mhr_model.character.parameter_transform.pose_parameters.sum() - 6)
    num_scale_params = int(mhr_model.character.parameter_transform.scaling_parameters.sum())

    lbs_parameter_names = mhr_model.character.parameter_transform.names[
        6: 6 + num_pose_param + num_scale_params
    ]

    # finger_parts = {"index", "middle", "ring", "pinky", "thumb"}
    # finger_parts = {"arm", "shoulder", "elbow"}
    foot_parts = {"foot", "toe", "ankle", "heel", "ball"}


    indices = []
    for i, name in enumerate(lbs_parameter_names[:num_pose_param]):
        if any(part in name.lower() for part in foot_parts):
            indices.append(i)

    return indices, lbs_parameter_names[:num_pose_param]


if __name__ == "__main__":
    indices, names = get_hand_pose_indices()

    print("Indices:")
    print(indices)
    print(len(indices))

    print("\nDetailed list:")
    for i in indices:
        print(f"{i}: {names[i]}")

