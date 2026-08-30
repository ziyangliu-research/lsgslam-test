"""TartanAir adapter for the released LSG-SLAM pose-graph + structure-refine backend.

This wrapper intentionally keeps tools/loop_closure/pose_graph_part_optim.py
unchanged. It applies only dataset/path routing patches at runtime:
  * TartanAir dataset adapter;
  * environment-driven base folder and sequence;
  * TartanAir RGB/depth loading during structure refinement.

The pose-graph, Gaussian deformation, and 5000-iteration structure-refinement
logic are the original released implementation.
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

    # Import the TartanAir adapter without changing the original backend file.
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
    from configs.tartanair.lsgslam import config
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

    module = types.ModuleType("lsg_tartanair_pose_graph_part_optim")
    module.__file__ = source_path
    module.__name__ = "__main__"
    exec(compile(source, source_path, "exec"), module.__dict__)


if __name__ == "__main__":
    main()
