import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

# enable .HEIC files
from utils.image import is_image, load_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('json_path')
    parser.add_argument('calib_path')

    return parser.parse_args()


def register_checkerboard_single_image(image, K, dist=None, debug=False, detect_max_side=1200):
    """Locate the 8x5 checkerboard, raise ValueError when there is none in the image.

    findChessboardCorners scales badly with resolution and is slowest exactly in the case we
    do not care about, an image without a board (tens of seconds on a 24 MP photo). The board
    is therefore searched on a downscaled copy, first with CALIB_CB_FAST_CHECK, and only the
    subpixel refinement of the corners runs at full resolution.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((5 * 8, 3), np.float32)
    objp[:, :2] = 0.03 * np.mgrid[0:8, 0:5].T.reshape(-1, 2)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, detect_max_side / max(gray.shape[:2]))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(small, (8, 5), flags | cv2.CALIB_CB_FAST_CHECK)
    if not ret:
        # the fast check gives up on boards it considers too weak, retry without it -- still
        # cheap, the full search only has to look at the downscaled image
        ret, corners = cv2.findChessboardCorners(small, (8, 5), flags)
    if not ret:
        raise ValueError("No checkerboard found")

    corners = (corners / scale).astype(np.float32)
    # The refinement window has to cover the error of the downscaled detection (a few pixels)
    # without reaching the neighbouring corner, so it is derived from the corner spacing.
    spacing = float(np.min(np.linalg.norm(np.diff(corners.reshape(5, 8, 2), axis=1), axis=-1)))
    win = int(np.clip(round(spacing / 3), 5, 30))
    corners2 = cv2.cornerSubPix(gray, corners, (win, win), (-1, -1), criteria)

    if debug:
        img = cv2.drawChessboardCorners(image, (8, 5), corners2, ret)
        cv2.imshow('img', cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4)))
        cv2.waitKey(1)

    return corners2, objp


def calibrate(images, debug=False):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((5 * 8, 3), np.float32)
    objp[:, :2] = 30 * np.mgrid[0:8, 0:5].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    shape = None

    for fname in tqdm(images):
        print(fname)
        img = load_image(fname)

        print(img.shape)
        if shape != img.shape and shape is not None:
            print(f"Warning: Image shape mismatch. Expected {shape}, got {img.shape}. Skipping {fname}")
            continue
        shape = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (8, 5), None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            if debug:
                img = cv2.drawChessboardCorners(img, (8, 5), corners2, ret)
                cv2.imshow('img', cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4)))
                cv2.waitKey(1)


    width = shape[1]
    height = shape[0]

    ret, mtx, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = \
        cv2.calibrateCameraExtended(objpoints, imgpoints, gray.shape[::-1], None, None)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print(f"total error: {mean_error / len(objpoints)} px")

    return mtx, dist, width, height, mean_error / len(objpoints)


def get_cam_dict(mtx, dist, width, height, err):
    d = {}
    d['focal'] = (mtx[0, 0] + mtx[1, 1]) / 2
    d['fx'] = mtx[0, 0]
    d['fy'] = mtx[1, 1]
    d['pp'] = mtx[:2, 2].tolist()
    d['K'] = mtx.tolist()
    d['fov'] = 2 * np.rad2deg(np.arctan(width / (2 * d['focal'])))
    d['distortion_coeffs'] = dist.tolist()
    d['width'] = width
    d['height'] = height
    d['mean_reprojection_error'] = err
    return d


def main(args):
    dir_path = args.calib_path
    images = [os.path.join(dir_path, x) for x in os.listdir(dir_path) if is_image(x)]
    print(dir_path)
    cam_dict = get_cam_dict(*calibrate(images, debug=args.debug))

    with open(args.json_path, 'w') as f:
        json.dump(cam_dict, f, indent=4)

    print(f"Calibration data saved to {args.json_path}")

if __name__ == '__main__':
    args = parse_args()
    main(args)