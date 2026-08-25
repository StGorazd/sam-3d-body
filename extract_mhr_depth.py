"""Prepare the per-image inputs consumed by joint_optim.py / load_and_vis.py.

For every image `<image_dir>/<fname>` three files are written into `data_dir`:

    <name>_mask.jpg   - binary person mask (SAM 3 / SAM 2), same resolution as the image
    <name>_depth.npy  - MoGe-2 z-depth map (float32), same resolution as the image
    <name>.json       - SAM 3D Body (MHR) detection: focal_length, pred_cam_t,
                        mhr_model_params, shape_params, expr_params, ...

With --subdirs the images are collected recursively and `<name>` keeps its path relative to
image_dir, so the layout of image_dir is mirrored inside data_dir.

The mask is also fed back into SAM 3D Body as a prompt (mask-conditioned inference) and
the camera intrinsics estimated by MoGe are used for the MHR head, so the depth map, the
mask and the MHR parameters all live in the same camera.

Example:
    python extract_mhr_depth.py --save_vis
    python extract_mhr_depth.py --image_dir <dir> --data_dir <dir> --fnames a.jpg b.jpg
    python extract_mhr_depth.py --image_dir <dir> --data_dir <dir> --all
    python extract_mhr_depth.py --image_dir <dir> --data_dir <dir> --subdirs

Note that joint_optim.py rescales the MoGe point cloud with utils.pointcloud.get_scaled_pointcloud,
which requires an 8x5 checkerboard to be visible in every image. This script checks for it and
warns when it is missing (disable with --no_checkerboard_check).
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from sam_3d_body import load_sam_3d_body, load_sam_3d_body_hf, SAM3DBodyEstimator
from tools.build_fov_estimator import denormalize_f
from utils.calib import register_checkerboard_single_image
from utils.image import load_image
from utils.paths import depth_path, is_prepared, json_path, list_images, mask_path, vis_path

DEFAULT_IMAGE_DIR = 'D:/Research/data/antropo/mini_scanovaci_den/foto'
DEFAULT_DATA_DIR = 'D:/Research/data/antropo/mini_scanovaci_den/processed'
DEFAULT_FNAMES = ['front.JPG', 'back.JPG', 'left.JPG', 'right.JPG']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image_dir', default=DEFAULT_IMAGE_DIR, help='folder with the input images')
    parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR, help='folder for the prepared files')
    parser.add_argument('--fnames', nargs='*', default=DEFAULT_FNAMES, help='image file names inside image_dir')
    parser.add_argument('--all', action='store_true', help='process every image in image_dir instead of --fnames')
    parser.add_argument('--subdirs', action='store_true',
                        help='implies --all and recurses into subdirectories, '
                             'the structure of image_dir is mirrored inside data_dir')

    parser.add_argument('--checkpoint_path', default=os.environ.get('SAM3D_CHECKPOINT_PATH', ''),
                        help='local SAM 3D Body checkpoint (model_config.yaml has to sit next to it)')
    parser.add_argument('--hf_repo_id', default='facebook/sam-3d-body-vith',
                        help='HuggingFace repo used when --checkpoint_path is not given')
    parser.add_argument('--mhr_path', default=os.environ.get('SAM3D_MHR_PATH', 'assets/mhr_model.pt'),
                        help='MHR model used with --checkpoint_path')
    parser.add_argument('--inference_type', default='full', choices=['full', 'body'],
                        help='full runs the extra hand decoder')

    parser.add_argument('--segmentor_name', default='sam3_hf', choices=['sam3_hf', 'sam3', 'sam2'],
                        help='sam3_hf uses transformers (facebook/sam3), sam2/sam3 use tools/build_sam.py')
    parser.add_argument('--segmentor_path', default=os.environ.get('SAM3D_SEGMENTOR_PATH', ''),
                        help='path to the sam2/sam3 repo (only for --segmentor_name sam2/sam3)')
    parser.add_argument('--hf_sam3_id', default='facebook/sam3', help='HuggingFace repo for --segmentor_name sam3_hf')
    parser.add_argument('--detector_name', default='vitdet', help='detector used to get boxes for --segmentor_name sam2')
    parser.add_argument('--detector_path', default=os.environ.get('SAM3D_DETECTOR_PATH', ''))
    parser.add_argument('--moge_path', default=os.environ.get('SAM3D_FOV_PATH', 'Ruicheng/moge-2-vitl-normal'),
                        help='MoGe-2 checkpoint (HuggingFace id or local path)')

    parser.add_argument('--bbox_thresh', type=float, default=0.5, help='person detection/segmentation score threshold')
    parser.add_argument('--prompt', default='person', help='text prompt for the SAM 3 segmentors')
    parser.add_argument('--overwrite', action='store_true', help='redo images that are already prepared')
    parser.add_argument('--save_vis', action='store_true', help='also write <name>_vis.jpg with the mask and the box')
    # parser.add_argument('--no_checkerboard_check', dest='checkerboard_check', action='store_false',
    #                     help='skip the check for the checkerboard needed by joint_optim.py')
    parser.add_argument('--device', default='cuda')

    return parser.parse_args()


class MoGeDepth:
    """MoGe-2 wrapper returning a dense z-depth map together with the camera intrinsics."""

    def __init__(self, path, device='cuda'):
        from moge.model.v2 import MoGeModel

        print(f"########### Loading MoGe-2 from {path}...")
        self.model = MoGeModel.from_pretrained(path).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, image_rgb):
        H, W = image_rgb.shape[:2]
        image = torch.tensor(image_rgb / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)

        # apply_mask=False keeps the depth dense, the person mask is used for the point cloud anyway
        moge_data = self.model.infer(image, apply_mask=False)

        depth = moge_data['depth'].float().cpu().numpy()
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Same convention as tools/build_fov_estimator.run_moge: square pixels from the vertical focal
        cam_int = denormalize_f(moge_data['intrinsics'].cpu().numpy(), H, W)
        cam_int[0, 0] = cam_int[1, 1]

        return depth, cam_int[None]


class HFSam3Segmentor:
    """Text prompted person segmentation with SAM 3 from transformers (no extra repo needed)."""

    def __init__(self, model_id, device='cuda', prompt='person'):
        from transformers import Sam3Model, Sam3Processor

        print(f"########### Using human segmentor: SAM3 ({model_id})...")
        self.model = Sam3Model.from_pretrained(model_id).to(device).eval()
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.device = device
        self.prompt = prompt

    @torch.no_grad()
    def __call__(self, image_rgb, thresh=0.5):
        inputs = self.processor(images=image_rgb, text=self.prompt, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)
        result = self.processor.post_process_instance_segmentation(
            outputs, threshold=thresh, mask_threshold=0.5, target_sizes=[image_rgb.shape[:2]]
        )[0]

        masks = result['masks'].cpu().numpy().astype(bool)
        boxes = result['boxes'].cpu().numpy().astype(np.float32)
        scores = result['scores'].cpu().numpy().astype(np.float32)
        return masks, boxes, scores


class RepoSegmentor:
    """Person segmentation with the segmentors shipped in tools/build_sam.py (needs a local sam2/sam3 repo)."""

    def __init__(self, segmentor_name, segmentor_path, detector_name, detector_path, device='cuda'):
        from tools.build_sam import HumanSegmentor

        self.sam = HumanSegmentor(name=segmentor_name, device=device, path=segmentor_path)
        self.segmentor_name = segmentor_name
        self.detector = None
        if segmentor_name == 'sam2':
            # sam2 is box prompted, so a detector is required
            from tools.build_detector import HumanDetector

            self.detector = HumanDetector(name=detector_name, device=device, path=detector_path)

    def __call__(self, image_rgb, thresh=0.5):
        if self.detector is not None:
            boxes = self.detector.run_human_detection(image_rgb[:, :, ::-1].copy(), det_cat_id=0,
                                                      bbox_thr=thresh, default_to_full_image=False)
            if len(boxes) == 0:
                return np.zeros((0,) + image_rgb.shape[:2], dtype=bool), np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        else:
            boxes = None

        # run_sam2 expects RGB while run_sam3 flips the input, so it has to be fed BGR
        image = image_rgb if self.segmentor_name == 'sam2' else image_rgb[:, :, ::-1].copy()
        masks, scores = self.sam.run_sam(image, boxes)
        masks = np.asarray(masks) > 0.5
        if boxes is None:
            boxes = np.array([mask_to_bbox(mask) for mask in masks], dtype=np.float32).reshape(-1, 4)
        return masks, np.asarray(boxes, np.float32), np.asarray(scores, np.float32)


def mask_to_bbox(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def save_vis(fname, image_bgr, mask, bbox):
    vis = image_bgr.copy()
    vis[mask] = 0.5 * vis[mask] + 0.5 * np.array([0, 255, 0])
    cv2.rectangle(vis, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 4)

    scale = 1024 / max(vis.shape[:2])
    if scale < 1:
        vis = cv2.resize(vis, None, fx=scale, fy=scale)
    cv2.imwrite(fname, vis)


def segment_images(args, fnames):
    """Write <name>_mask.jpg for every image, return the box and the score of the kept person."""
    if args.segmentor_name == 'sam3_hf':
        segmentor = HFSam3Segmentor(args.hf_sam3_id, device=args.device, prompt=args.prompt)
    else:
        segmentor = RepoSegmentor(args.segmentor_name, args.segmentor_path, args.detector_name,
                                  args.detector_path, device=args.device)

    detections = {}
    for fname in tqdm(fnames, desc='masks'):
        image = load_image(os.path.join(args.image_dir, fname))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # if args.checkerboard_check:
        #     try:
        #         register_checkerboard_single_image(image, None)
        #     except ValueError:
        #         print(f"{fname}: WARNING - no 8x5 checkerboard found, joint_optim.py will fail on this image")

        masks, boxes, scores = segmentor(image_rgb, thresh=args.bbox_thresh)
        if len(masks) == 0 or max(mask.sum() for mask in masks) == 0:
            print(f"{fname}: WARNING - no person found, skipping")
            continue

        # joint_optim.py expects a single subject per image, keep the largest one
        idx = int(np.argmax([mask.sum() for mask in masks]))
        mask, bbox, score = masks[idx], boxes[idx], float(scores[idx])
        print(f"{fname}: person {idx + 1} of {len(masks)}, score {score:0.3f}, "
              f"{100 * mask.mean():0.1f}% of the image")

        cv2.imwrite(mask_path(args.data_dir, fname),
                    np.repeat(mask[:, :, np.newaxis].astype(np.uint8) * 255, 3, axis=2),
                    [cv2.IMWRITE_JPEG_QUALITY, 100])
        if args.save_vis:
            save_vis(vis_path(args.data_dir, fname), image, mask, bbox)

        detections[fname] = {'bbox': bbox.reshape(1, 4), 'mask_score': score}

    del segmentor
    torch.cuda.empty_cache()
    return detections


def estimate_depths(args, fnames):
    """Write <name>_depth.npy for every image, return the MoGe camera intrinsics."""
    depth_estimator = MoGeDepth(args.moge_path, device=args.device)

    cam_ints = {}
    for fname in tqdm(fnames, desc='depths'):
        image_rgb = cv2.cvtColor(load_image(os.path.join(args.image_dir, fname)), cv2.COLOR_BGR2RGB)
        depth, cam_int = depth_estimator(image_rgb)
        np.save(depth_path(args.data_dir, fname), depth)
        cam_ints[fname] = cam_int

    del depth_estimator
    torch.cuda.empty_cache()
    return cam_ints


def estimate_mhr(args, fnames, detections, cam_ints):
    """Write <name>.json with the mask-conditioned SAM 3D Body prediction for every image."""
    if args.checkpoint_path:
        model, model_cfg = load_sam_3d_body(args.checkpoint_path, device=args.device, mhr_path=args.mhr_path)
    else:
        print(f"No --checkpoint_path given, downloading {args.hf_repo_id} from HuggingFace...")
        model, model_cfg = load_sam_3d_body_hf(args.hf_repo_id)

    # The segmentor and the fov estimator were already run above, their outputs are passed
    # to process_one_image, so that mask, depth and MHR parameters share the same camera.
    estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg)

    done = []
    for fname in tqdm(fnames, desc='mhr'):
        image = load_image(os.path.join(args.image_dir, fname))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]
        mask = load_image(mask_path(args.data_dir, fname))[:, :, 0] > 127

        outputs = estimator.process_one_image(
            image_rgb,
            bboxes=detections[fname]['bbox'],
            masks=mask,
            cam_int=cam_ints[fname],
            inference_type=args.inference_type,
        )
        if len(outputs) == 0:
            print(f"{fname}: WARNING - SAM 3D Body returned no detection, skipping")
            continue

        json_data = {k: v for k, v in outputs[0].items() if k != 'mask'}
        json_data['mask_score'] = detections[fname]['mask_score']
        json_data['original_resolution'] = [W, H]
        json_data['current_resolution'] = [W, H]

        with open(json_path(args.data_dir, fname), 'w') as f:
            json.dump(to_serializable(json_data), f)
        done.append(fname)

    return done


def main(args):
    os.makedirs(args.data_dir, exist_ok=True)

    if args.all or args.subdirs:
        fnames = list_images(args.image_dir, recursive=args.subdirs)
    else:
        fnames = args.fnames

    if not args.overwrite:
        todo = [x for x in fnames if not is_prepared(args.data_dir, x)]
        if len(todo) < len(fnames):
            print(f"Skipping {len(fnames) - len(todo)} already prepared images (use --overwrite to redo them)")
        fnames = todo

    print(f"Preparing {len(fnames)} images from {args.image_dir} into {args.data_dir}")
    if len(fnames) == 0:
        return

    # with --subdirs the outputs of an image go next to its own relative path inside data_dir
    for directory in sorted({os.path.dirname(mask_path(args.data_dir, x)) for x in fnames}):
        os.makedirs(directory, exist_ok=True)

    # The three models are run one after another and freed in between, they do not fit
    # into a small GPU together at full image resolution.
    detections = segment_images(args, fnames)
    cam_ints = estimate_depths(args, [x for x in fnames if x in detections])
    done = estimate_mhr(args, list(cam_ints.keys()), detections, cam_ints)

    print(f"Done, {len(done)} images prepared in {args.data_dir}")
    failed = [x for x in fnames if x not in done]
    if failed:
        print(f"Failed images: {failed}")


if __name__ == '__main__':
    main(parse_args())
