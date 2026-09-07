import os

# Final EuRoC paper-benchmark configuration.
# Keep the released LSG-SLAM EuRoC hyperparameters/calibration, but evaluate the
# full downloaded sequence with temporal stride=5 and a strict 8:2 holdout on
# the retained stream.  Retained sample ids 4,9,14,... are pose-only.

primary_device = "cuda:0"
seed = 0

scene_name = os.environ.get("EUROC_SEQUENCE", "MH_02_easy")
cam0_dir = os.environ.get("EUROC_CAM0_DIR", f"euroc/{scene_name}/mav0/cam0")
workdir = os.environ.get("LSG_WORKDIR", "experiments/euroc_official_full_stride5_final4")

map_every = 1
keyframe_every = 1
mapping_window_size = 24
tracking_iters = 100
mapping_iters = 100
run_loop_closure = True

image_width = 752
image_height = 480

start_idx = int(os.environ.get("EUROC_START", "0"))
end_idx = int(os.environ.get("EUROC_END", "-1"))
stride = int(os.environ.get("EUROC_STRIDE", "5"))

# Split is defined AFTER temporal downsampling.  With sequence start=0 and
# stride=5, retained sample ids 4,9,14,... correspond to raw dataset indices
# 20,45,70,... .  The split runner/backend use this raw-index representation.
eval_split_every = 25
eval_split_offset = 20

run_name = f"{scene_name}_{start_idx}_{end_idx}_{stride}"
group_name = "euroc_stride5_final4"

config = dict(
    workdir=workdir,
    run_name=run_name,
    scene_path=os.path.join(workdir, run_name, "params.npz"),
    seed=seed,
    primary_device=primary_device,
    map_every=map_every,
    keyframe_every=keyframe_every,
    mapping_window_size=mapping_window_size,
    report_global_progress_every=500,
    eval_every=1,
    eval_split_every=eval_split_every,
    eval_split_offset=eval_split_offset,
    scene_radius_depth_ratio=3,
    mean_sq_dist_method="projective",
    gaussian_distribution="isotropic",
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False,
    checkpoint_interval=100,
    use_warp_loss=True,
    weight_warp=100,
    use_grad_mask=False,
    opt_local_map=False,
    run_loop_closure=run_loop_closure,
    use_wandb=False,
    pixel_gs_depth_gamma=0.37,
    wandb=dict(
        entity="",
        project="",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        basedir=cam0_dir,
        gradslam_data_cfg="./configs/euroc/euroc.yaml",
        sequence=scene_name,
        desired_image_height=image_height,
        desired_image_width=image_width,
        start=start_idx,
        end=end_idx,
        stride=stride,
        num_frames=-1,
    ),
    tracking=dict(
        use_gt_poses=False,
        forward_prop=True,
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        icp_corr_threshold=0.5,
        loss_weights=dict(im=1.0, depth=0.2),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.0004,
            cam_trans=0.002,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5,
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(im=0.5, depth=1.0),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0,
            cam_trans=0.0,
        ),
        prune_gaussians=True,
        pruning_dict=dict(
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=500,
        ),
        use_gaussian_splatting_densification=False,
        densify_dict=dict(
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000,
        ),
    ),
    viz=dict(
        render_mode="color",
        offset_first_viz_cam=True,
        show_sil=False,
        visualize_cams=True,
        viz_w=2560,
        viz_h=1600,
        viz_near=0.01,
        viz_far=100.0,
        view_scale=2,
        viz_fps=5,
        enter_interactive_post_online=True,
    ),
)
