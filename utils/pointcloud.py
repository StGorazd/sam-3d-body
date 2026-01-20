import itertools

import numpy as np

from utils.calib import register_checkerboard_single_image


def load_moge(path):
    ...


def unproject_points(x, K):
    # params: x - Nx2 array of points
    #         K - 3x3 intrinsics matrix
    #         output - Nx3 array of unprojected 3d points
    K_inv = np.linalg.inv(K)
    x_hom = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    x_3d = (K_inv @ x_hom.T).T
    return x_3d

def get_moge_pointcloud(moge_depth, K):
    # params: moge_depth - HxW array of depth values
    #         K - 3x3 intrinsics matrix
    #         output - Nx3 array of 3d points

    # Get image coordinates
    H, W = moge_depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    uv = np.stack((u.flatten(), v.flatten()), axis=1)

    points_3d_unscaled = unproject_points(uv, K)

    points_3d = points_3d_unscaled * moge_depth.flatten()[:, np.newaxis]
    # points_3d = points_3d[moge_depth.flatten() > 0]

    return points_3d.reshape(H, W, 3)


def get_opt_scale(corners, objp, points3d_moge, dim = 30):
    moge_corners = []
    used_corners = []
    for corner in corners:
        # calculate the linear interpolation of the points3d_moge using the subpixel corner coordinates
        x, y = corner[0]
        x1, y1 = int(x), int(y)
        x2, y2 = x1 + 1, y1 + 1

        # Bilinear interpolation
        if 0 <= x1 < points3d_moge.shape[1] and 0 <= y1 < points3d_moge.shape[0]:
            p1 = points3d_moge[y1, x1]
            p2 = points3d_moge[y1, x2]
            p3 = points3d_moge[y2, x1]
            p4 = points3d_moge[y2, x2]

            wx2 = x - x1
            wx1 = 1.0 - wx2
            wy2 = y - y1
            wy1 = 1.0 - wy2

            # Interpolate
            interpolated_point = (wx1 * wy1 * p1 + wx2 * wy1 * p2 + wx1 * wy2 * p3 + wx2 * wy2 * p4)
            moge_corners.append(interpolated_point)
            used_corners.append(corner)

    # now calculate all combinations of relative distances of all pairs of points in corners and do the same for objp
    moge_distances = []
    for p1, p2 in itertools.combinations(moge_corners, 2):
        moge_distances.append(np.linalg.norm(p1 - p2))

    objp_distances = []
    for p1, p2 in itertools.combinations(objp, 2):
        objp_distances.append(np.linalg.norm(p1 - p2))

    moge_distances = np.array(moge_distances)
    objp_distances = np.array(objp_distances)
    scale = np.median(objp_distances / moge_distances)

    # L2 version
    # scale = np.sum(objp_distances * moge_distances) / np.sum(moge_distances**2)
    return scale


def get_scaled_pointcloud(image, moge_depth, K):
    corners, objp = register_checkerboard_single_image(image, K)
    points3d_moge = get_moge_pointcloud(moge_depth, K)

    scale = get_opt_scale(corners, objp, points3d_moge)
    return scale * points3d_moge