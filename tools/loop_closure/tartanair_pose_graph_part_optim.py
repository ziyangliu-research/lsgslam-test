"""TartanAir adapter for the released LSG-SLAM pose-graph + structure-refine backend.

The released tools/loop_closure/pose_graph_part_optim.py stays unchanged on disk.
This wrapper applies only TartanAir routing and the experiment protocol at
runtime. For the 8:2 benchmark, test frames (global ids 4,9,14,...) may take
part in pose estimation / loop closure, but they are excluded from the 5000-
iteration structure-refinement loss. Final rendering is evaluated on every
frame and reported separately for train/test.

The wrapper also records optimization-only backend timing (PGO, Gaussian
deformation, and SR), explicitly excluding before/after metric rendering.
"""

import os
import sys
import types


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _BASE_DIR)


def _replace_once(source, old, new, label):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Could not apply TartanAir backend patch '{label}': expected 1 match, found {count}. "
            "The released pose_graph_part_optim.py may have changed."
        )
    return source.replace(old, new, 1)


def main():
    source_path = os.path.join(_BASE_DIR, "tools", "loop_closure", "pose_graph_part_optim.py")
    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    source = _replace_once(
        source,
        "import numpy as np\n",
        "import numpy as np\nimport json\nimport time\n",
        "json/time import",
    )

    source = _replace_once(
        source,
        "from pytorch_msssim import ms_ssim\n",
        "from datasets.gradslam_datasets.tartanair import TartanAirDataset\nfrom pytorch_msssim import ms_ssim\n",
        "dataset import",
    )

    source = _replace_once(
        source,
        '''def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["kitti"]:
        return KittiDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["euroc"]:
        return EurocDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")
''',
        '''def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["kitti"]:
        return KittiDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["euroc"]:
        return EurocDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tartanair", "tartan"]:
        return TartanAirDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")
''',
        "dataset factory",
    )

    source = _replace_once(
        source,
        '''    base_folder = 'MH_01_easy' # path to folder with loop closure results
    scene_name = 'MH_01_easy'
    dataset_type = 'euroc' # kitti or euroc
    if dataset_type == 'kitti':
        kitti_base_folder = '' 
        image_folder_path = os.path.join(kitti_base_folder, scene_name, 'image_2')
        depth_folder_path = os.path.join(kitti_base_folder, scene_name, 'depth_sceneflow')
    elif dataset_type == 'euroc':
        from configs.euroc.lsgslam import config
        dataset_config = config["data"]
        dataset_config['basedir'] = f"euroc/{scene_name}/mav0/cam0"
        dataset_config['sequence'] = scene_name
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
''',
        '''    base_folder = os.environ["LSG_FULL_BASE_FOLDER"]
    scene_name = os.environ["TARTANAIR_SEQUENCE"]
    dataset_type = 'tartanair'
    from configs.tartanair.lsgslam_full_split_8_2 import config
    dataset_config = config["data"]
    dataset_config['basedir'] = os.path.join(
        os.environ.get(
            "TARTANAIR_DATA_ROOT",
            "/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge",
        ),
        "stereo",
    )
    dataset_config['sequence'] = scene_name
    gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
''',
        "main dataset routing",
    )

    source = _replace_once(
        source,
        '''        elif dataset_type == 'euroc':
            euroc_dataset = get_dataset(
                config_dict=gradslam_data_cfg,
                basedir=dataset_config["basedir"],
                sequence=os.path.basename(dataset_config["sequence"]),
                start=start_idx,
                end=end_idx,
                stride=stride,
                desired_height=dataset_config["desired_image_height"],
                desired_width=dataset_config["desired_image_width"],
                device='cuda',
                relative_pose=True,
            )
            gt_imgs, depths = load_imgs_from_path_list(euroc_dataset.color_paths, euroc_dataset.depth_paths)
''',
        '''        elif dataset_type == 'euroc':
            euroc_dataset = get_dataset(
                config_dict=gradslam_data_cfg,
                basedir=dataset_config["basedir"],
                sequence=os.path.basename(dataset_config["sequence"]),
                start=start_idx,
                end=end_idx,
                stride=stride,
                desired_height=dataset_config["desired_image_height"],
                desired_width=dataset_config["desired_image_width"],
                device='cuda',
                relative_pose=True,
            )
            gt_imgs, depths = load_imgs_from_path_list(euroc_dataset.color_paths, euroc_dataset.depth_paths)
        elif dataset_type == 'tartanair':
            tartanair_dataset = get_dataset(
                config_dict=gradslam_data_cfg,
                basedir=dataset_config["basedir"],
                sequence=os.path.basename(dataset_config["sequence"]),
                start=start_idx,
                end=end_idx,
                stride=stride,
                desired_height=dataset_config["desired_image_height"],
                desired_width=dataset_config["desired_image_width"],
                device='cuda',
                relative_pose=True,
            )
            gt_imgs, depths = load_imgs_from_path_list(
                tartanair_dataset.color_paths, tartanair_dataset.depth_paths
            )
''',
        "structure-refine image loading",
    )

    source = _replace_once(
        source,
        '''    before_opt_psnr_list = []
    before_opt_ssim_list = []
    before_opt_lpips_list = []

    after_opt_psnr_list = []
    after_opt_ssim_list = []
    after_opt_lpips_list = []
''',
        '''    before_opt_psnr_list = []
    before_opt_ssim_list = []
    before_opt_lpips_list = []

    after_opt_psnr_list = []
    after_opt_ssim_list = []
    after_opt_lpips_list = []

    before_train_psnr, before_train_ssim = [], []
    before_test_psnr, before_test_ssim = [], []
    after_train_psnr, after_train_ssim = [], []
    after_test_psnr, after_test_ssim = [], []
    total_gaussians = 0
    pgo_opt_seconds = 0.0
    gaussian_deformation_seconds = 0.0
    sr_opt_seconds = 0.0
''',
        "split metric and timing accumulators",
    )

    source = _replace_once(
        source,
        '''            PGM.optimizePoseGraph()
''',
        '''            _pgo_t0 = time.perf_counter()
            PGM.optimizePoseGraph()
            pgo_opt_seconds += time.perf_counter() - _pgo_t0
''',
        "PGO timing",
    )

    source = _replace_once(
        source,
        '''                before_opt_psnr_list.append(psnr.cpu().numpy())
                before_opt_ssim_list.append(ssim.cpu().numpy())
                before_opt_lpips_list.append(lpips_score)
''',
        '''                before_opt_psnr_list.append(psnr.cpu().numpy())
                before_opt_ssim_list.append(ssim.cpu().numpy())
                before_opt_lpips_list.append(lpips_score)
                ssim_single = calc_ssim(weighted_im, weighted_gt_im).mean()
                global_frame_idx = start_idx + i * stride
                if global_frame_idx % 5 == 4:
                    before_test_psnr.append(float(psnr.detach().cpu()))
                    before_test_ssim.append(float(ssim_single.detach().cpu()))
                else:
                    before_train_psnr.append(float(psnr.detach().cpu()))
                    before_train_ssim.append(float(ssim_single.detach().cpu()))
''',
        "before-SR split metrics",
    )

    source = _replace_once(
        source,
        '''        # deformation
        pre_est_w2cs = torch.tensor(np.array(before_opt_w2cs), dtype=torch.float32, device='cuda') # (k, 4, 4)
''',
        '''        # deformation
        torch.cuda.synchronize()
        _deform_t0 = time.perf_counter()
        pre_est_w2cs = torch.tensor(np.array(before_opt_w2cs), dtype=torch.float32, device='cuda') # (k, 4, 4)
''',
        "deformation timing start",
    )

    source = _replace_once(
        source,
        '''        params['means3D'] = torch.nn.Parameter(gs_new_means3d[:, :3].cuda().float().contiguous().requires_grad_(True))


        # save the pc when need to debug
''',
        '''        params['means3D'] = torch.nn.Parameter(gs_new_means3d[:, :3].cuda().float().contiguous().requires_grad_(True))
        torch.cuda.synchronize()
        gaussian_deformation_seconds += time.perf_counter() - _deform_t0


        # save the pc when need to debug
''',
        "deformation timing end",
    )

    source = _replace_once(
        source,
        '''        for i in tqdm(range(structure_refine_total_iters), 'Color refining...'):

            index = random.randint(0, optim_c2ws.shape[0]-1)
''',
        '''        train_local_indices = [
            idx for idx in range(optim_c2ws.shape[0])
            if ((start_idx + idx * stride) % 5) != 4
        ]
        if len(train_local_indices) == 0:
            raise RuntimeError(f"No training frames available for SR part {start_idx}..{end_idx}")
        print(f"[Split-SR] train={len(train_local_indices)}/{optim_c2ws.shape[0]} frames")
        torch.cuda.synchronize()
        _sr_t0 = time.perf_counter()
        for i in tqdm(range(structure_refine_total_iters), 'Color refining...'):

            index = random.choice(train_local_indices)
''',
        "train-only structure refinement and timing start",
    )

    source = _replace_once(
        source,
        '''            # TODO  lr update

        # eval structure refine
''',
        '''            # TODO  lr update

        torch.cuda.synchronize()
        sr_opt_seconds += time.perf_counter() - _sr_t0

        # eval structure refine
''',
        "SR timing end",
    )

    source = _replace_once(
        source,
        '''                after_opt_psnr_list.append(psnr.cpu().numpy())
                after_opt_ssim_list.append(ssim.cpu().numpy())
                after_opt_lpips_list.append(lpips_score)
''',
        '''                after_opt_psnr_list.append(psnr.cpu().numpy())
                after_opt_ssim_list.append(ssim.cpu().numpy())
                after_opt_lpips_list.append(lpips_score)
                ssim_single = calc_ssim(weighted_im, weighted_gt_im).mean()
                global_frame_idx = start_idx + i * stride
                if global_frame_idx % 5 == 4:
                    after_test_psnr.append(float(psnr.detach().cpu()))
                    after_test_ssim.append(float(ssim_single.detach().cpu()))
                else:
                    after_train_psnr.append(float(psnr.detach().cpu()))
                    after_train_ssim.append(float(ssim_single.detach().cpu()))
''',
        "after-SR split metrics",
    )

    source = _replace_once(
        source,
        '''        save_params(params, rendering_save_dir)
        print(pixel_gs_depth_gamma*pixel_gs_scene_radius)
''',
        '''        total_gaussians += int(params['means3D'].shape[0])
        save_params(params, rendering_save_dir)
        print(pixel_gs_depth_gamma*pixel_gs_scene_radius)
''',
        "Gaussian count accumulation",
    )

    source = _replace_once(
        source,
        '''    with open(os.path.join(rendering_save_dir, 'avg_metrics.txt'), 'w', encoding='utf-8') as file:
''',
        '''    gt_xyz = gt_traj_pts.T.astype(np.float64)
    est_xyz = loop_traj_pts.T.astype(np.float64)
    valid_pose = np.isfinite(gt_xyz).all(axis=1) & np.isfinite(est_xyz).all(axis=1)
    gt_valid = gt_xyz[valid_pose]
    est_valid = est_xyz[valid_pose]
    if len(gt_valid) >= 3:
        gt_mean = gt_valid.mean(axis=0)
        est_mean = est_valid.mean(axis=0)
        X = est_valid - est_mean
        Y = gt_valid - gt_mean
        U, _, Vt = np.linalg.svd(X.T @ Y)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = gt_mean - R @ est_mean
        est_aligned = (R @ est_valid.T).T + t
        ate_rmse = float(np.sqrt(np.mean(np.sum((est_aligned - gt_valid) ** 2, axis=1))))
    else:
        ate_rmse = float('nan')

    def _mean_or_nan(values):
        return float(np.mean(values)) if len(values) else float('nan')

    split_summary = {
        'sequence': scene_name,
        'maxmap_ratio': float(valid_pose.sum() / len(valid_pose)),
        'tracked_frames': int(valid_pose.sum()),
        'num_frames': int(len(valid_pose)),
        'ate_rmse_se3_m': ate_rmse,
        'before_sr_train_psnr': _mean_or_nan(before_train_psnr),
        'before_sr_train_ssim': _mean_or_nan(before_train_ssim),
        'before_sr_test_psnr': _mean_or_nan(before_test_psnr),
        'before_sr_test_ssim': _mean_or_nan(before_test_ssim),
        'after_sr_train_psnr': _mean_or_nan(after_train_psnr),
        'after_sr_train_ssim': _mean_or_nan(after_train_ssim),
        'after_sr_test_psnr': _mean_or_nan(after_test_psnr),
        'after_sr_test_ssim': _mean_or_nan(after_test_ssim),
        'gaussians': int(total_gaussians),
        'ssim_type': 'single-scale SSIM',
        'split_rule': 'global frames 4,9,14,... are pose-only; excluded from mapping and SR loss',
        'fps': None,
    }
    split_summary_path = os.path.join(base_folder, 'benchmark_summary_full_split.json')
    with open(split_summary_path, 'w', encoding='utf-8') as split_file:
        json.dump(split_summary, split_file, indent=2)

    backend_timing = {
        'pgo_seconds': float(pgo_opt_seconds),
        'gaussian_deformation_seconds': float(gaussian_deformation_seconds),
        'structure_refinement_seconds': float(sr_opt_seconds),
        'backend_optimization_seconds': float(pgo_opt_seconds + gaussian_deformation_seconds + sr_opt_seconds),
        'scope': 'PGO optimizer calls + Gaussian deformation + 5000-iter SR; before/after metric rendering excluded',
    }
    backend_timing_path = os.path.join(base_folder, 'backend_optimization_timing.json')
    with open(backend_timing_path, 'w', encoding='utf-8') as timing_file:
        json.dump(backend_timing, timing_file, indent=2)

    print()
    print('================ Full LSG-SLAM 8:2 Summary ================')
    print(f"Sequence:             {scene_name}")
    print(f"MaxMap:               {100.0 * split_summary['maxmap_ratio']:.2f}%")
    print(f"ATE RMSE (SE3):       {ate_rmse:.6f} m")
    print(f"Train PSNR/SSIM:      {split_summary['after_sr_train_psnr']:.4f} / {split_summary['after_sr_train_ssim']:.6f}")
    print(f"Test PSNR/SSIM:       {split_summary['after_sr_test_psnr']:.4f} / {split_summary['after_sr_test_ssim']:.6f}")
    print(f"Gaussians:            {total_gaussians}")
    print(f"Backend opt time:     {backend_timing['backend_optimization_seconds']:.3f} s")
    print(f"Summary:              {split_summary_path}")
    print(f"Timing:               {backend_timing_path}")
    print('===========================================================')
    print()

    with open(os.path.join(rendering_save_dir, 'avg_metrics.txt'), 'w', encoding='utf-8') as file:
''',
        "full split summary and backend timing",
    )

    module = types.ModuleType("lsg_tartanair_pose_graph_part_optim")
    module.__file__ = source_path
    module.__name__ = "__main__"
    exec(compile(source, source_path, "exec"), module.__dict__)


if __name__ == "__main__":
    main()
