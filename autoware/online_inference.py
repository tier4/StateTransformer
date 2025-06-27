import sys
sys.path.append("/home/acf15382lp/projects/STR2/rasterize_wrapper")
sys.path.append("/home/acf15382lp/projects/Autoware-Python-ROSBAG-Loader")
from rasterize_wrapper.rasterize_online import RasterizeWrapper
import numpy as np
import torch
from datasets import Dataset, Features, Array3D, Array2D, Sequence, Value
from tqdm import tqdm
import os
from core.trajectory_loader import ExceedFrameRangeError
from train import AutowareCollator
from transformer4planning.models.backbone.str_base import build_models, build_model_from_path
from torch.utils.data import DataLoader
from PIL import Image
import copy
import cv2
import matplotlib.pyplot as plt
import matplotlib
import io

# figureの警告を抑制し、メモリ使用量を制御
matplotlib.rcParams['figure.max_open_warning'] = 50  # 警告の閾値を上げる

def visualize_output(inputs, outputs, j):
    """
    Rasterize形式での可視化を行う関数
    """
    high_res_raster = inputs["high_res_raster"][j].cpu().numpy()
    low_res_raster = inputs["low_res_raster"][j].cpu().numpy()
    context_actions = inputs["context_actions"][j].cpu().numpy()
    trajectory_label = inputs["trajectory_label"][j].cpu().numpy()
    high_res_raster = high_res_raster.transpose(2, 0, 1)
    low_res_raster = low_res_raster.transpose(2, 0, 1)
    trajectory_prediction = outputs["traj_logits"][j].detach().cpu().numpy()
    # ego_info = inputs["ego_info"][0].cpu().numpy()
    # ego_yaw = ego_info[2]
    # bbox_length = ego_info[3]
    # bbox_width = ego_info[4]
    bbox_length = 4.0
    bbox_width = 2.0
    ego_x = 0
    ego_y = 0
    range_m = 30
    high_res_raster = high_res_raster * 255.0
    low_res_raster = low_res_raster * 255.0
    trajectory_label[..., :2] = trajectory_label[..., :2] * 100.0
    trajectory_prediction[..., :2] = trajectory_prediction[..., :2] * 100.0

    total_result = np.zeros((3, 224, 224), dtype=np.uint8)
    ego_color = np.array([0, 0, 255], dtype=np.uint8)
    other_color = np.array([0, 255, 0], dtype=np.uint8)
    img = low_res_raster
    route = img[0]
    road = img[2]
    ego = img[26:26 + 4]
    ego = ego[-1]
    other = img[26 + 4:26 + 4 + 4]
    other = other[-1]

    total_img = np.where((route + road) > 0, 255, 0)
    total_result[0] = total_img
    total_result[1] = total_img
    total_result[2] = total_img
    total_result[0] = np.where((ego + total_img) > 0, 255, 0)
    total_result[1] = np.where((other + total_img) > 0, 255, 0)

    def visualize_trajectory(trajectory, img, range_m, color_channel=0):

        def _world_to_image(x, y, range_m):
            x_norm = (x + range_m) / (2 * range_m)
            y_norm = (y + range_m) / (2 * range_m)
            x_img = int(x_norm * 224)
            y_img = int(y_norm * 224)
            return x_img, 224 - y_img
        
        img_coords = []
        
        for t in range(len(trajectory)):
            x, y, yaw = trajectory[t][0], trajectory[t][1], trajectory[t][3]
            x_img, y_img = _world_to_image(x, y, range_m)
            img_coords.append((x_img, y_img))

        cv2.polylines(img[color_channel], [np.array(img_coords)], isClosed=False, color=255, thickness=2)
        
    visualize_trajectory(trajectory_label, total_result, range_m, color_channel=0)
    visualize_trajectory(trajectory_prediction, total_result, range_m, color_channel=2)
    total_result = Image.fromarray(total_result.transpose(1, 2, 0))
    return total_result

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


def matplotlib_to_pil_image(ax: plt.Axes) -> Image.Image:
    """
    matplotlibのaxをPIL Imageに変換する関数
    """
    # axの親figureを取得

    ax.set_axis_off()
    fig = ax.figure
    
    # canvasのバッファからRGBデータを取得
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    
    return img

def visualize_matplotlib(inputs, outputs, ax: plt.Axes, j):
    """
    matplotlibのaxを使用して可視化を行う関数
    """
    high_res_raster = inputs["high_res_raster"][j].cpu().numpy()
    low_res_raster = inputs["low_res_raster"][j].cpu().numpy()
    context_actions = inputs["context_actions"][j].cpu().numpy()
    trajectory_label = inputs["trajectory_label"][j].cpu().numpy()
    high_res_raster = high_res_raster.transpose(2, 0, 1)
    low_res_raster = low_res_raster.transpose(2, 0, 1)
    trajectory_prediction = outputs["traj_logits"][j].detach().cpu().numpy()
    trajectory_prediction[..., :2] = trajectory_prediction[..., :2] * 100.0
    trajectory_label[..., :2] = trajectory_label[..., :2] * 100.0

    # 回転行列を適用 (ego_yaw の逆回転)
    def _world_to_image(x, y, range_m, width=800, height=800):
        x_coords = []
        y_coords = []

        for x, y in zip(x, y):
            dx = x
            dy = y

            # 回転行列適用 (自車両が常に上を向くようにする)
            x_rot = x
            y_rot = y

            # 正規化（ワールド座標 -> [0, 1] のスケール）
            x_norm = (x_rot + range_m) / (2 * range_m)
            y_norm = (y_rot + range_m) / (2 * range_m)

            # 画像座標に変換
            x_img = int(x_norm * width)
            y_img = int(y_norm * height)

            # 画像のY座標は上下が反転しているため補正
            x_coords.append(x_img)
            y_coords.append(y_img)

        return x_coords, y_coords

    x_coords, y_coords, _, _ = zip(*trajectory_label)
    x_coords, y_coords = _world_to_image(x_coords, y_coords, range_m=30)
    ax.plot(x_coords, y_coords, color='green', linestyle='-', linewidth=2)
    x_coords, y_coords, _, _ = zip(*trajectory_prediction)
    x_coords, y_coords = _world_to_image(x_coords, y_coords, range_m=30)
    ax.plot(x_coords, y_coords, color='purple', linestyle='-', linewidth=2)

    ax.text(5, 15, 'Ground Truth', color='green', fontsize=12, fontweight='bold', backgroundcolor='white')
    ax.text(5, 50, 'Prediction', color='purple', fontsize=12, fontweight='bold', backgroundcolor='white')
    ax.text(5, 85, 'Ego', color='red', fontsize=12, fontweight='bold', backgroundcolor='white')
    ax.text(5, 120, 'Other', color='blue', fontsize=12, fontweight='bold', backgroundcolor='white')

    return ax
    



def create_online_dataset(osm_file_path, db_path):

    rasterizer = RasterizeWrapper(osm_file_path=osm_file_path, dbdir=db_path)

    i = 3
    datas = []
    axes = []
    while True:
        try:

            fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
            ax.set_xlim(0, 800)
            ax.set_ylim(0, 800)
            dummy_fig, dummy_ax = plt.subplots(figsize=(8, 8), dpi=100)
            high_res_range_m = 10
            low_res_range_m = 30
            high_other_trajectory_imgs, high_ego_trajectory_imgs, high_context_actions, high_trajectory_labels, high_ego_info, dummy_ax = rasterizer.rasterize_rosbag(keyframe=i, before_keyframe=3, after_keyframe=80, range_m=high_res_range_m, ax=dummy_ax)
            high_res_map_imgs, dummy_ax = rasterizer.rasterize_map(range_m=high_res_range_m, ax=dummy_ax)
            low_other_trajectory_imgs, low_ego_trajectory_imgs, low_context_actions, low_trajectory_labels, low_ego_info, ax = rasterizer.rasterize_rosbag(keyframe=i, before_keyframe=3, after_keyframe=80, range_m=low_res_range_m, ax=ax)
            low_res_map_imgs, ax = rasterizer.rasterize_map(range_m=low_res_range_m, ax=ax)

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
            
            # dummy_figは不要なので閉じる
            plt.close(dummy_fig)
            
            i += 1

            datas.append(data)
            axes.append(ax)

        except ExceedFrameRangeError:
            break

    dataset = Dataset.from_list(datas)
    return dataset, axes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    osm_file_path = "/home/acf15382lp/projects/datasets/lanelet2_map.osm"
    db_path = "/home/acf15382lp/Downloads/rosbag-data/34434667-2a91-4b84-b5f2-d68b46d04f46"
    dataset, axes = create_online_dataset(osm_file_path, db_path)
    print(dataset)

    collator = AutowareCollator(device=device)

    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collator, shuffle=False)

    path = "/groups/gcd50654/tier4/kai-yamashita/full-scratch-mixtral-800m/output/checkpoint-5000"
    model = build_model_from_path(model_path=path)
    model = model.from_pretrained(path)
    model.eval()
    model.to(device)

    frames = []
    i = 0

    for batch in tqdm(dataloader, desc="Generating frames"):
        with torch.no_grad():
            outputs = model.forward(**batch)
        print(len(batch))
        for j in range(len(batch)):
            ax = visualize_matplotlib(batch, outputs["pred_dict"], axes[i], j)
            # frame = visualize_output(batch, outputs["pred_dict"], j)
            frame = matplotlib_to_pil_image(ax)
            frames.append(frame)
        
            # figureを閉じてメモリを解放
            plt.close(axes[i].figure)
            i += 1
        
    frames[0].save("output.gif", save_all=True, append_images=frames[1:], duration=200, loop=0)

        

if __name__ == "__main__":
    main()