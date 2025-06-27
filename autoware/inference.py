
import copy
import torch
import copy
import numpy as np
import imageio
from transformer4planning.models.backbone.str_base import build_model_from_path
from torch.utils.data import DataLoader
from datasets import Dataset, load_from_disk
from train import AutowareCollator
import cv2
from PIL import Image, ImageDraw, ImageFont


def visualize_text(img, text, position):
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    draw = ImageDraw.Draw(img)
    draw.text(position, text, fill=(255, 255, 255), font=font)
    return img


def concatenate_gifs(gif1_path, gif2_path, output_path):
    """
    2つのGIF画像を水平方向に連結して新しいGIFを作成する関数
    
    Parameters:
    gif1_path (str): 上側に配置するGIF画像のパス
    gif2_path (str): 下側に配置するGIF画像のパス
    output_path (str): 出力先のGIFファイルパス
    """
    # 両方のGIFを読み込む
    gif1_frames = imageio.mimread(gif1_path)
    gif2_frames = imageio.mimread(gif2_path)
    print(np.array(gif1_frames).shape)
    print(np.array(gif2_frames).shape)
    print(len(gif1_frames))
    print(len(gif2_frames))
    
    # フレーム数を合わせる（少ない方に合わせる）
    frame_count = min(len(gif1_frames), len(gif2_frames))
    
    # 連結したフレームを格納するリスト
    concatenated_frames = []
    
    # 各フレームを水平方向に連結
    for i in range(frame_count):
        # PILのImageオブジェクトに変換
        img1 = Image.fromarray(gif1_frames[i])
        img2 = Image.fromarray(gif2_frames[i])
        
        # サイズを確認して必要なら224x224にリサイズ
        if img1.size != (224, 224):
            img1 = img1.resize((224, 224))
        if img2.size != (224, 224):
            img2 = img2.resize((224, 224))
        
        # 新しい画像を作成（横幅224、高さ448）
        new_img = Image.new('RGBA', (224, 448))
        
        # 2つの画像を配置
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (0, 224))
        
        # 配列に変換して追加
        concatenated_frames.append(np.array(new_img))
    
    # 結果を保存
    imageio.mimsave(output_path, concatenated_frames, format='GIF', duration=1/30, loop=0)  # フレームレートは適宜調整
    
    print(f"連結GIFを作成しました: {output_path}")
    
def _world_length_to_pixel(length_m, range_m, width):
    """
    ワールド座標系での長さ（メートル）をピクセル画像上での長さに変換

    Parameters:
    length_m (float): 変換したい長さ（メートル）
    range_m (float): 画像の表示範囲（メートル）

    Returns:
    int: ピクセル単位での長さ
    """
    # range_mがピクセル画像の半分の幅に対応
    # 例：range_m=10mの場合、20m×20mの領域を画像化
    pixels_per_meter = width / (2 * range_m)
    return int(length_m * pixels_per_meter)

def trajectory_visualize(trajectory, ego_x, ego_y, ego_yaw, bbox_length, bbox_width, range_m):

    def _world_to_image(x, y, range_m):
        # 自車両基準の相対座標
        dx = x - ego_x
        dy = y - ego_y

        # 回転行列を適用 (ego_yaw の逆回転)
        cos_theta = np.cos(-ego_yaw)
        sin_theta = np.sin(-ego_yaw)

        # 回転行列適用 (自車両が常に上を向くようにする)
        x_rot = cos_theta * dx - sin_theta * dy
        y_rot = sin_theta * dx + cos_theta * dy

        # 正規化（ワールド座標 -> [0, 1] のスケール）
        x_norm = (x_rot + range_m) / (2 * range_m)
        y_norm = (y_rot + range_m) / (2 * range_m)

        # 画像座標に変換
        x_img = int(x_norm * 224)
        y_img = int(y_norm * 224)

        # 画像のY座標は上下が反転しているため補正
        return x_img, 224 - y_img
    ego_agent_channel = np.zeros((len(trajectory), 224, 224), dtype=np.uint8)
    for t in range(len(trajectory)):
        x, y, yaw = trajectory[t][0], trajectory[t][1], trajectory[t][3]
        yaw = yaw + ego_yaw
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        x_modified = cos_yaw * x - sin_yaw * y
        y_modified = sin_yaw * x + cos_yaw * y
        x, y = x_modified + ego_x, y_modified + ego_y
        x_img, y_img = _world_to_image(x, y, range_m)
        bbox_width_pixel = _world_length_to_pixel(bbox_width, range_m, 224)
        bbox_length_pixel = _world_length_to_pixel(bbox_length, range_m, 224)
        _draw_rotated_box(ego_agent_channel[t], x_img, y_img, bbox_width_pixel, bbox_length_pixel, yaw - ego_yaw)
    
    return ego_agent_channel

def _draw_rotated_box(img, center_x, center_y, width, length, yaw):
        """
        回転した長方形を描画する関数
        
        Parameters:
        img: 描画対象の画像
        center_x, center_y: ボックスの中心座標
        width: ボックスの幅
        length: ボックスの長さ
        yaw: ボックスの回転角（ラジアン）
        """
        # 長方形の4つの頂点を計算
        half_length = length / 2
        half_width = width / 2
        
        # 回転行列
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # 4つの頂点の相対座標
        corners = np.array([
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ])
        
        # 回転を適用
        rotated_corners = np.array([
            [cos_yaw * x - sin_yaw * y + center_x,
            sin_yaw * x + cos_yaw * y + center_y]
            for x, y in corners
        ], dtype=np.int32)

        # 塗りつぶし
        cv2.fillPoly(img, [rotated_corners], color=255)

def visualize_output(inputs, outputs, output_path):
    high_res_raster = inputs["high_res_raster"][0].cpu().numpy()
    low_res_raster = inputs["low_res_raster"][0].cpu().numpy()
    context_actions = inputs["context_actions"][0].cpu().numpy()
    trajectory_label = inputs["trajectory_label"][0].cpu().numpy()
    high_res_raster = high_res_raster.transpose(2, 0, 1)
    low_res_raster = low_res_raster.transpose(2, 0, 1)
    trajectory_prediction = outputs["pred_dict"]["traj_logits"][0].detach().cpu().numpy()
    # ego_info = inputs["ego_info"][0].cpu().numpy()
    # ego_yaw = ego_info[2]
    # bbox_length = ego_info[3]
    # bbox_width = ego_info[4]
    bbox_length = 4.0
    bbox_width = 2.0
    ego_yaw = 0
    ego_x = 0
    ego_y = 0
    print(f"bbox_length: {bbox_length}, bbox_width: {bbox_width}")
    print(f"ego_yaw: {ego_yaw}")
    range_m = 30
    high_res_raster = high_res_raster * 255.0
    low_res_raster = low_res_raster * 255.0
    trajectory_label[..., :2] = trajectory_label[..., :2] * 100.0
    trajectory_prediction[..., :2] = trajectory_prediction[..., :2] * 100.0

    total_result = np.zeros((3, 224, 224), dtype=np.uint8)
    ego_color = np.array([0, 0, 255], dtype=np.uint8)
    other_color = np.array([0, 255, 0], dtype=np.uint8)
    for i, img in enumerate([high_res_raster, low_res_raster]):
        route = img[0]
        road = img[2]
        ego = img[26:26 + 4]
        other = img[26 + 4:26 + 4 + 4]

        

        # print(f"max: {route.max()}, min: {route.min()}")
        # print(f"max: {road.max()}, min: {road.min()}")

        # # Save as PNG
        total_img = np.where((route + road) > 0, 255, 0)
        total_result[0] = total_img
        total_result[1] = total_img
        total_result[2] = total_img
        # name = "high_res" if i == 0 else "low_res"
        # cv2.imwrite(f"{name}_route.png", route)
        # cv2.imwrite(f"{name}_road.png", road)
        # cv2.imwrite(f"{name}_total.png", total_img)
        # # egoとotherはgifにして保存
        # ego_frames = []
        # other_frames = []

        total_frames = []
        for f in range(len(ego)):
            total_frame = copy.deepcopy(total_result)
            ego_frame = ego[f]
            total_frame[0] = np.where((ego_frame + total_img) > 0, 255, 0)
            # ego_frame = Image.fromarray(ego_frame)
            # ego_frames.append(ego_frame)
            other_frame = other[f]
            # other_frame = Image.fromarray(other_frame)
            # other_frames.append(other_frame)
            total_frame[1] = np.where((other_frame + total_img) > 0, 255, 0)
            total_frame = Image.fromarray(total_frame.transpose(1, 2, 0))
            total_frame = visualize_text(total_frame, f"his. frame {f}", (10, 10))
            total_frames.append(total_frame)
        # ego_frames[0].save(f"{name}_ego.gif", save_all=True, append_images=ego_frames[1:], duration=200, loop=0)
        # other_frames[0].save(f"{name}_other.gif", save_all=True, append_images=other_frames[1:], duration=200, loop=0)
        # total_frames[0].save(f"high_res_total.gif", save_all=True, append_images=total_frames[1:], duration=200, loop=0)

    ego_agent_channel = trajectory_visualize(trajectory_label, ego_x, ego_y, ego_yaw, bbox_length, bbox_width, range_m)
    prediction_channel = trajectory_visualize(trajectory_prediction, ego_x, ego_y, ego_yaw, bbox_length, bbox_width, range_m)
    traj_frames = []
    # for f in range(len(ego_agent_channel)):
    #     traj_frame = Image.fromarray(ego_agent_channel[f])
    #     traj_frames.append(traj_frame)
    # traj_frames[0].save("label_trajectory.gif", save_all=True, append_images=traj_frames[1:], duration=200, loop=0)

    # traj_frames = []
    # for f in range(len(prediction_channel)):
    #     traj_frame = Image.fromarray(prediction_channel[f])
    #     traj_frames.append(traj_frame)
    # traj_frames[0].save("prediction_trajectory.gif", save_all=True, append_images=traj_frames[1:], duration=200, loop=0)

    for f in range(len(ego_agent_channel)):
        total_frame = copy.deepcopy(total_result)
        total_frame[0] = np.where((ego_agent_channel[f] + total_img) > 0, 255, 0)
        total_frame[2] = np.where((prediction_channel[f] + total_img) > 0, 255, 0)
        total_frame = Image.fromarray(total_frame.transpose(1, 2, 0))
        total_frame = visualize_text(total_frame, f"pred. frame {f}", (10, 10))
        total_frames.append(total_frame)
    total_frames[0].save(f"{output_path}.gif", save_all=True, append_images=total_frames[1:], duration=200, loop=0)

    # concatenate_gifs("label_trajectory.gif", "prediction_trajectory.gif", "concatenated_trajectory.gif")
    

    
    
    


def main():
    path = "/groups/gcd50654/tier4/kai-yamashita/full-scratch-mixtral-800m/output/checkpoint-3000"
    model = build_model_from_path(model_path=path)
    model = model.from_pretrained(path)
    model.eval()
    model.to("cuda")

    # データセットをロード
    dataset = load_from_disk("/groups/gcd50654/tier4/dataset/rosbag_full_dataset")
    collator = AutowareCollator(device="cuda")

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=True)
    for i in range(10):
        inputs = next(iter(dataloader))
        outputs = model.forward(**inputs)
        visualize_output(inputs, outputs, f"trajectory_comparison_{i}")


def measure_time():
    import time
    import torch
    import numpy as np
    
    path = "/home/acf15382lp/autoware_2024/output/checkpoint-1000"
    model = build_model_from_path(model_path=path)
    model = model.from_pretrained(path)
    model.eval()
    model.to("cuda")

    dataset = load_from_disk("/home/acf15382lp/projects/STR2/rasterize_wrapper/dataset")
    collator = AutowareCollator(device="cuda")

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=True)
    inputs = next(iter(dataloader))
    
    # ウォームアップ実行
    with torch.no_grad():
        _ = model.forward(**inputs)
    
    # 推論時間の測定
    iterations = 100
    inference_times = []
    
    for _ in range(iterations):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            _ = model.forward(**inputs)
            
        torch.cuda.synchronize()
        end_time = time.time()
        inference_times.append(end_time - start_time)
    
    # 結果の表示
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    print(f"推論時間の測定結果（{iterations}回の平均）:")
    print(f"平均時間: {avg_time*1000:.2f} ms")
    print(f"標準偏差: {std_time*1000:.2f} ms")
    print(f"最小時間: {min_time*1000:.2f} ms")
    print(f"最大時間: {max_time*1000:.2f} ms")
    print(f"推論速度: {1.0/avg_time:.2f} FPS")
    
    
    
    

if __name__ == "__main__":
    main()