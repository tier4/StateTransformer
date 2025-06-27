from rasterize_class import RasterizeWrapper
import numpy as np
import torch
from datasets import Dataset, Features, Array3D, Array2D, Sequence, Value
from tqdm import tqdm
import os
import sys
sys.path.append("/home/acf15382lp/projects/Autoware-Python-ROSBAG-Loader")
from utils.generate_metadata import extract_metadata_from_directory
from core.trajectory_loader import ExceedFrameRangeError


def process_imgs(map_imgs, other_trajectory_imgs, ego_trajectory_imgs):
    ego_road_imgs = map_imgs['ego_road']
    other_road_imgs = map_imgs['other_road']

    # 疑似データを追加してegoのchannel数を2に, otherのchannel数を20に
    ego_road_imgs = np.stack([ego_road_imgs, np.zeros_like(ego_road_imgs)], axis=0)
    other_road_imgs = np.concatenate([other_road_imgs[np.newaxis, :, :], np.zeros((19, other_road_imgs.shape[0], other_road_imgs.shape[1]))], axis=0)
    dummy_traffic_sign_imgs = np.zeros((4, 224, 224))

    trajectory_imgs = np.concatenate([ego_trajectory_imgs, other_trajectory_imgs], axis=0)
    trajectory_imgs = np.concatenate([trajectory_imgs, np.zeros((24, 224, 224))], axis=0)

    total_imgs = np.concatenate([ego_road_imgs, other_road_imgs, dummy_traffic_sign_imgs, trajectory_imgs], axis=0)
    total_imgs = np.transpose(total_imgs, (1, 2, 0))
    assert total_imgs.shape == (224, 224, 58), f"total_imgs.shape: {total_imgs.shape}"
    return total_imgs





def main(osm_file_path, db_path_list, dataset_dir):
    def generator(db_path_list):
        for db_path in db_path_list:
            if not os.path.isfile(os.path.join(db_path, 'metadata.yaml')):
                extract_metadata_from_directory(db_path) # dbdirにmetadata.yamlを自動生成
            rasterizer = RasterizeWrapper(osm_file_path=osm_file_path, dbdir=db_path)
            i = 3
            while True:
                try:
                    high_res_range_m = 10
                    low_res_range_m = 30
                    high_other_trajectory_imgs, high_ego_trajectory_imgs, high_context_actions, high_trajectory_labels, high_ego_info = rasterizer.rasterize_rosbag(keyframe=i, before_keyframe=3, after_keyframe=80, range_m=high_res_range_m)
                    high_res_map_imgs = rasterizer.rasterize_map(range_m=high_res_range_m)
                    low_other_trajectory_imgs, low_ego_trajectory_imgs, low_context_actions, low_trajectory_labels, low_ego_info = rasterizer.rasterize_rosbag(keyframe=i, before_keyframe=3, after_keyframe=80, range_m=low_res_range_m)
                    low_res_map_imgs = rasterizer.rasterize_map(range_m=low_res_range_m)

                    ego_info = np.array(high_ego_info).astype(np.float32)

                    high_res_imgs = process_imgs(high_res_map_imgs, high_other_trajectory_imgs, high_ego_trajectory_imgs)
                    low_res_imgs = process_imgs(low_res_map_imgs, low_other_trajectory_imgs, low_ego_trajectory_imgs)
                    high_context_actions = np.array(high_context_actions)
                    high_trajectory_labels = np.array(high_trajectory_labels)

                    assert high_context_actions.shape == (4, 4), f"high_context_actions.shape: {high_context_actions.shape}"
                    assert high_trajectory_labels.shape == (80, 4), f"low_trajectory_labels.shape: {high_trajectory_labels.shape}"

                    data = {
                        "high_res_raster": np.asarray(high_res_imgs).astype(np.uint8),
                        "low_res_raster": np.asarray(low_res_imgs).astype(np.uint8),
                        "context_actions": np.asarray(high_context_actions).astype(np.float32),
                        "trajectory_label": np.asarray(high_trajectory_labels).astype(np.float32),
                        "ego_info": np.asarray(high_ego_info).astype(np.float32),
                    }
                    i += 1
                    yield data
                except ExceedFrameRangeError:
                    break
                except Exception as e:
                    print(f"Error at {db_path} {i}: {e}")
                    # エラー内容をファイルに保存
                    with open(f"./error_logs.txt", "a") as f:
                        f.write(f"Error at {db_path} {i}: {e}\n")
                    break

    features = Features({
        "high_res_raster": Array3D(dtype="uint8", shape=(224, 224, 58)),
        "low_res_raster" : Array3D(dtype="uint8", shape=(224, 224, 58)),
        "context_actions": Array2D(dtype="float32", shape=(4, 4)),
        "trajectory_label": Array2D(dtype="float32", shape=(80, 4)),
        "ego_info": Sequence(feature=Value(dtype="float32"))
    })


    dataset = Dataset.from_generator(generator, gen_kwargs={"db_path_list": db_path_list}, num_proc=10, features=features, writer_batch_size=200)
    # Save To Local
    os.makedirs(dataset_dir, exist_ok=True)
    dataset.save_to_disk(dataset_dir, max_shard_size="2GB")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--osm_file_path", type=str, default="/home/acf15382lp/projects/datasets/lanelet2_map.osm")
    parser.add_argument("--db_list_path_file", type=str, default="/home/acf15382lp/Downloads/rosbag-data/db_list.txt")
    parser.add_argument("--dataset_dir", type=str, default="/groups/gcd50654/tier4/kai-yamashita/autoware_dataset")
    args = parser.parse_args()

    osm_file_path = args.osm_file_path
    db_list_path_file = args.db_list_path_file
    with open(db_list_path_file, "r") as f:
        db_list = f.readlines()
        # 改行を削除
        db_list = [db_path.strip() for db_path in db_list]

    main(osm_file_path=args.osm_file_path, db_path_list=db_list, dataset_dir=args.dataset_dir)



    



