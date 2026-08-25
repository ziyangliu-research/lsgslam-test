#!/usr/bin/env python3
"""Precompute IGEV-Stereo depth and TransVPR global features for TartanAir V1."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
IGEV_ROOT = ROOT / "third_party" / "IGEV-Stereo"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(IGEV_ROOT))

from core.igev_stereo import IGEVStereo  # noqa: E402
from core.utils.utils import InputPadder  # noqa: E402
from third_party.TransVPR.blocks import POOL  # noqa: E402
from third_party.TransVPR.feature_extractor import Extractor_base  # noqa: E402


DEFAULT_DATA_ROOT = Path(
    "/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge"
)
ALL_SEQUENCES = [f"SE{i:03d}" for i in range(8)] + [
    f"SH{i:03d}" for i in range(8)
]

FX = 320.0
BASELINE = 0.25
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640


def _load_torch_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_igev(device, checkpoint_path):
    args = SimpleNamespace(
        mixed_precision=False,
        valid_iters=32,
        hidden_dims=[128, 128, 128],
        corr_implementation="reg",
        shared_backbone=False,
        corr_levels=2,
        corr_radius=4,
        n_downsample=2,
        slow_fast_gru=False,
        n_gru_layers=3,
        max_disp=192,
    )

    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[device.index or 0])
    state_dict = _load_torch_checkpoint(checkpoint_path)
    model.load_state_dict(state_dict)
    model = model.module.to(device)
    model.eval()
    return model, args


def _build_transvpr(device, checkpoint_path):
    model = Extractor_base()
    pool = POOL(model.embedding_dim)
    model.add_module("pool", pool)
    state_dict = _load_torch_checkpoint(checkpoint_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize([IMAGE_HEIGHT, IMAGE_WIDTH]),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return model, transform


def _load_rgb_tensor(path, device):
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float()[None]
    return tensor.to(device)


def _select_frame_ids(sequence_dir, start, end):
    left_dir = sequence_dir / "image_left"
    right_dir = sequence_dir / "image_right"

    left_paths = sorted(left_dir.glob("*_left.png"))
    if not left_paths:
        raise FileNotFoundError(f"No TartanAir left images found in {left_dir}")

    selected = []
    for left_path in left_paths:
        frame_id = left_path.name.split("_")[0]
        frame_idx = int(frame_id)
        if frame_idx < start:
            continue
        if end is not None and frame_idx > end:
            continue

        right_path = right_dir / f"{frame_id}_right.png"
        if not right_path.is_file():
            raise FileNotFoundError(
                f"Missing right image for frame {frame_id}: {right_path}"
            )
        selected.append((frame_id, left_path, right_path))

    if not selected:
        raise ValueError(
            f"No frames selected in {sequence_dir} for start={start}, end={end}."
        )
    return selected


def process_sequence(
    sequence,
    data_root,
    device,
    igev_model,
    igev_args,
    vpr_model,
    vpr_transform,
    make_depth=True,
    make_features=True,
    start=0,
    end=None,
    overwrite=False,
):
    sequence_dir = data_root / "stereo" / sequence
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"TartanAir sequence not found: {sequence_dir}")

    frames = _select_frame_ids(sequence_dir, start, end)
    depth_dir = sequence_dir / "depth_sceneflow"
    feature_dir = sequence_dir / "global_features"
    depth_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[{sequence}] {len(frames)} frames, raw range "
        f"{frames[0][0]}..{frames[-1][0]}"
    )

    with torch.inference_mode():
        for frame_id, left_path, right_path in tqdm(
            frames, desc=f"{sequence}", unit="frame"
        ):
            depth_out = depth_dir / f"{frame_id}.npy"
            feature_out = feature_dir / f"{frame_id}.npy"

            if make_depth and (overwrite or not depth_out.exists()):
                left = _load_rgb_tensor(left_path, device)
                right = _load_rgb_tensor(right_path, device)

                padder = InputPadder(left.shape, divis_by=32)
                left_pad, right_pad = padder.pad(left, right)

                disparity = igev_model(
                    left_pad,
                    right_pad,
                    iters=igev_args.valid_iters,
                    test_mode=True,
                )
                disparity = padder.unpad(disparity).squeeze().float()
                disparity_np = disparity.detach().cpu().numpy()

                valid = np.isfinite(disparity_np) & (disparity_np > 1e-6)
                depth = np.zeros_like(disparity_np, dtype=np.float32)
                depth[valid] = (FX * BASELINE) / disparity_np[valid]
                depth[~np.isfinite(depth)] = 0.0
                depth[depth < 0.1] = 0.0
                np.save(depth_out, depth)

            if make_features and (overwrite or not feature_out.exists()):
                image = Image.open(left_path).convert("RGB")
                image_tensor = vpr_transform(image)[None].to(device)
                patch_feat = vpr_model(image_tensor)
                global_feat, _ = vpr_model.pool(patch_feat)
                np.save(
                    feature_out,
                    global_feat.detach().cpu().numpy()[0].astype(np.float32),
                )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate the depth_sceneflow and global_features folders used by "
            "LSG-SLAM for TartanAir V1 stereo challenge sequences."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="TartanAir_Stereo_Challenge root directory.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=ALL_SEQUENCES,
        help="Sequence names. Default: SE000-SE007 and SH000-SH007.",
    )
    parser.add_argument("--start", type=int, default=0, help="First raw frame index.")
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last raw frame index, inclusive. Default: process to the end.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--igev-checkpoint",
        type=Path,
        default=IGEV_ROOT / "pretrained_models" / "sceneflow.pth",
    )
    parser.add_argument(
        "--vpr-checkpoint",
        type=Path,
        default=ROOT / "third_party" / "TransVPR" / "TransVPR_MSLS.pth",
    )
    parser.add_argument(
        "--depth-only",
        action="store_true",
        help="Generate only IGEV-Stereo depth.",
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Generate only TransVPR global features.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute files that already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.depth_only and args.features_only:
        raise ValueError("--depth-only and --features-only are mutually exclusive.")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("LSG-SLAM TartanAir preprocessing requires a CUDA GPU.")

    make_depth = not args.features_only
    make_features = not args.depth_only

    print(f"Data root: {args.data_root}")
    print(f"Device: {device}")
    print(f"Sequences: {', '.join(args.sequences)}")
    print(f"Depth: {make_depth}, global features: {make_features}")

    igev_model = None
    igev_args = None
    if make_depth:
        igev_model, igev_args = _build_igev(device, args.igev_checkpoint)

    vpr_model = None
    vpr_transform = None
    if make_features:
        vpr_model, vpr_transform = _build_transvpr(device, args.vpr_checkpoint)

    for sequence in args.sequences:
        if sequence not in ALL_SEQUENCES:
            raise ValueError(
                f"Unsupported challenge sequence {sequence}. "
                f"Expected one of: {', '.join(ALL_SEQUENCES)}"
            )
        process_sequence(
            sequence=sequence,
            data_root=args.data_root,
            device=device,
            igev_model=igev_model,
            igev_args=igev_args,
            vpr_model=vpr_model,
            vpr_transform=vpr_transform,
            make_depth=make_depth,
            make_features=make_features,
            start=args.start,
            end=args.end,
            overwrite=args.overwrite,
        )

    print("TartanAir preprocessing complete.")


if __name__ == "__main__":
    main()
