import lanelet2
from lanelet2.core import Point2d, BoundingBox2d, BasicPoint2d, LineString3d
from lanelet2.io import load, Origin
from lanelet2.projection import UtmProjector
import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
sys.path.append("/home/acf15382lp/projects/Autoware-Python-ROSBAG-Loader")
from core.trajectory_loader import load_trajectory

from PIL import Image
import glob
import os
import re

def create_gif_from_images(input_folder, output_filename, pattern="*.png", duration=200):
    """
    指定されたフォルダ内の画像からGIFアニメーションを作成する
    
    Parameters:
    input_folder (str): 入力画像が含まれるフォルダのパス
    output_filename (str): 出力するGIFファイルの名前
    pattern (str): 画像ファイルのパターン（デフォルト: "rosbag_*.png"）
    duration (int): 各フレームの表示時間（ミリ秒）
    """
    # 画像ファイルのパスを取得
    image_files = glob.glob(os.path.join(input_folder, pattern))
    
    # ファイル名から数字を抽出してソート
    def extract_number(filename):
        return int(re.search(r'\d+', os.path.basename(filename)).group())
    
    image_files.sort(key=extract_number)
    
    # 最初の画像を開いてGIFアニメーションの準備
    frames = []
    for image_file in image_files:
        frame = Image.open(image_file)
        frames.append(frame)
    
    # GIFアニメーションを保存
    frames[0].save(
        output_filename,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    
    print(f"GIFアニメーションを作成しました: {output_filename}")

def save_channels_as_png(rasterized_map, name_prefix, output_dir):
    """
    各チャンネルをPNGファイルとして保存する関数

    Parameters:
    rasterized_map (numpy.ndarray): ラスタライズされた地図データ (n_channels, width, height)
    output_dir (str): PNGファイルを保存するディレクトリ
    name_prefix (str): 保存するPNGファイルの名前のprefix
    """
    # ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    # 各チャンネルを保存
    for i, channel in enumerate(rasterized_map):
        cv2.imwrite(os.path.join(output_dir, f"{name_prefix}_{i}.png"), channel)

class RasterizeWrapper:

    def __init__(
        self,
        osm_file_path: str,
        dbdir: str,
        origin_point: tuple = (35.2231249, 138.8024583),
        num_pixels: int = 224
    ):
        self.origin_point = origin_point
        self.projector = UtmProjector(Origin(self.origin_point[0], self.origin_point[1]))
        self.lanelet_map = load(osm_file_path, self.projector)

        self.num_pixels = num_pixels
        self.height, self.width = num_pixels, num_pixels

        self.dbdir = dbdir
        self.ego_x = None
        self.ego_y = None
        self.ego_yaw = None
        self.ego_bbox_length = None
        self.ego_bbox_width = None
        self.ego_bbox_height = None
    
    def _world_to_image(self, x, y, range_m, width=None, height=None):
        # 自車両基準の相対座標
        dx = x - self.ego_x
        dy = y - self.ego_y

        if width is None:
            reverse_y = True
        else:
            reverse_y = False

        if width is None:
            width = self.width
        if height is None:
            height = self.height

        # 回転行列を適用 (ego_yaw の逆回転)
        cos_theta = np.cos(-self.ego_yaw)
        sin_theta = np.sin(-self.ego_yaw)

        # 回転行列適用 (自車両が常に上を向くようにする)
        x_rot = cos_theta * dx - sin_theta * dy
        y_rot = sin_theta * dx + cos_theta * dy

        # 正規化（ワールド座標 -> [0, 1] のスケール）
        x_norm = (x_rot + range_m) / (2 * range_m)
        y_norm = (y_rot + range_m) / (2 * range_m)

        # 画像座標に変換
        x_img = int(x_norm * width)
        y_img = int(y_norm * height)

        # 画像のY座標は上下が反転しているため補正
        if reverse_y:
            return x_img, height - y_img
        else:
            return x_img, y_img
    
    def _world_length_to_pixel(self, length_m, range_m, width=None):
        if width is None:
            width = self.width
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

    def rasterize_map(self, range_m, ax: plt.Axes):
        assert self.ego_x is not None
        assert self.ego_y is not None
        assert self.ego_yaw is not None
        assert self.ego_bbox_length is not None
        assert self.ego_bbox_width is not None
        assert self.ego_bbox_height is not None
        # チャンネルの初期化をまとめる
        channels = {
            'ego_road': np.zeros((self.height, self.width), dtype=np.uint8),
            'other_road': np.zeros((self.height, self.width), dtype=np.uint8),
            'traffic_sign': np.zeros((self.height, self.width), dtype=np.uint8),
            'traffic_light': np.zeros((self.height, self.width), dtype=np.uint8),
            'regulatory_element': np.zeros((self.height, self.width), dtype=np.uint8),
            'right_of_way': np.zeros((self.height, self.width), dtype=np.uint8)
        }

        # Egoレーンの検出
        ego_position = BasicPoint2d(self.ego_x, self.ego_y)
        nearest_ls = lanelet2.geometry.findNearest(self.lanelet_map.laneletLayer, ego_position, 1)
        ego_lanelet = nearest_ls[0][1] if nearest_ls else None


        # レーンの描画
        def draw_lanelet(lanelet, is_ego=False):
            channel = channels['ego_road'] if is_ego else channels['other_road']
            bounds = [
                [self._world_to_image(pt.x, pt.y, range_m) for pt in lanelet.leftBound],
                [self._world_to_image(pt.x, pt.y, range_m) for pt in lanelet.rightBound]
            ]
            for bound in bounds:
                cv2.polylines(channel, [np.array(bound)], isClosed=False, color=255, thickness=2)
            
            if lanelet.centerline is not None:
                centerline = [self._world_to_image(pt.x, pt.y, range_m) for pt in lanelet.centerline]
                cv2.polylines(channel, [np.array(centerline)], isClosed=False, color=255, thickness=1)

        # レーンレットの処理
        for lanelet in self.lanelet_map.laneletLayer:
            draw_lanelet(lanelet, is_ego=(lanelet == ego_lanelet))

            bounds = [
                [self._world_to_image(pt.x, pt.y, range_m, width=800, height=800) for pt in lanelet.leftBound],
                [self._world_to_image(pt.x, pt.y, range_m, width=800, height=800) for pt in lanelet.rightBound]
            ]
            for bound in bounds:
                x_coords, y_coords = zip(*bound)
                ax.plot(x_coords, y_coords, color='black', linestyle='-', linewidth=2)

            if lanelet.centerline is not None:
                centerline = [self._world_to_image(pt.x, pt.y, range_m, width=800, height=800) for pt in lanelet.centerline]
                x_coords, y_coords = zip(*centerline)
                ax.plot(x_coords, y_coords, color='black', linestyle='--', linewidth=1)
        
    



        # ポリゴンの描画
        def draw_polygon(points, channel, closed=True):
            coords = []
            for pt in points:
                try:
                    # まずattributesから'local_x'と'local_y'を取得しようとする
                    x = float(pt.attributes['local_x'])
                    y = float(pt.attributes['local_y'])
                except (KeyError, ValueError):
                    # 取得できない場合は直接x, y座標を使用
                    x, y = pt.x, pt.y
                
                dpt = lanelet2.core.BasicPoint2d(x, y)
                x_img, y_img = self._world_to_image(dpt.x, dpt.y, range_m)
                coords.append([x_img, y_img])
            
            if coords:
                cv2.polylines(channel, [np.array(coords)], isClosed=closed, color=255, thickness=2)

        # ポリゴンレイヤーの処理
        for polygon in self.lanelet_map.polygonLayer:
            draw_polygon(polygon, channels['right_of_way'])

        # 規制要素の処理
        def process_regulatory_element(element):
            for param_list in element.parameters.values():
                for param in param_list:
                    if isinstance(param, lanelet2.core.ConstLanelet):
                        draw_lanelet(param)
                    elif isinstance(param, lanelet2.core.ConstLineString3d):
                        points = []
                        for pt in param:
                            x, y = float(pt.attributes['local_x']), float(pt.attributes['local_y'])
                            dpt = lanelet2.core.BasicPoint2d(x, y)
                            point = self._world_to_image(dpt.x, dpt.y, range_m)
                            cv2.circle(channels['regulatory_element'], point, radius=5, color=255, thickness=-1)
                            points.append(point)
                        if points:
                            cv2.polylines(channels['regulatory_element'], [np.array(points)], isClosed=False, color=255, thickness=2)

        for element in self.lanelet_map.regulatoryElementLayer:
            process_regulatory_element(element)

        # チャンネルをスタックして出力
        # return np.stack(list(channels.values()), axis=0)
        return channels, ax
    def rasterize_rosbag(self, keyframe, before_keyframe, after_keyframe, range_m, ax: plt.Axes):
        # trajectory:  dict{'uuid1': [[timestamp, x, y, z, yaw, bbox_length, bbox_width, bbox_height], [timestamp, x, y, z, yaw, bbox_length, bbox_width, bbox_height], ...], 
        #                   'uuid2': [[timestamp, x, y, z, yaw, bbox_length, bbox_width, bbox_height], [timestamp, x, y, z, yaw, bbox_length, bbox_width, bbox_height], ...],
        # egoのデータは最近傍時間のフレームにダウンサンプルして他objectのtrajectoryフレームに投影．
        # ego_trajectory = trajectory["ego"]
        trajectories, reference = load_trajectory(dbdir=self.dbdir, 
                                                keyframe=keyframe, 
                                                before_keyframe=before_keyframe,
                                                after_keyframe=after_keyframe, 
                                                reference_output=True)
        
        # reference_outputフラグでkeyframeのegoの絶対座標を追加出力 [x, y, z, yaw, bbox_length, bbox_width, bbox_height]
        # trajectoryは変わらずkeyframeのegoからの相対座標

        self.ego_x, self.ego_y, self.ego_yaw, self.ego_bbox_length, self.ego_bbox_width, self.ego_bbox_height = reference[0], reference[1], reference[3], reference[4], reference[5], reference[6]
        ego_trajectory = trajectories["ego"]
        trajectory_length = len(trajectories["ego"])
        # Other Agent の描画
        other_agent_channels = []
        ego_agent_channels = []
        context_actions = []
        trajectory_labels = []
        for t in range(trajectory_length):
            if t <= before_keyframe:
                other_agent_channel = np.zeros((self.height, self.width), dtype=np.uint8)
                ego_agent_channel = np.zeros((self.height, self.width), dtype=np.uint8)
                for name in trajectories.keys():
                    if name == "ego":
                        trajectory = trajectories[name]
                        x, y, yaw, bbox_length, bbox_width, bbox_height = trajectory[t][1], trajectory[t][2], trajectory[t][4], trajectory[t][5], trajectory[t][6], trajectory[t][7]
                        cos_yaw = np.cos(-self.ego_yaw)
                        sin_yaw = np.sin(-self.ego_yaw)
                        x_rot = cos_yaw * x - sin_yaw * y
                        y_rot = sin_yaw * x + cos_yaw * y
                        x, y = x + self.ego_x, y + self.ego_y
                        x_img, y_img = self._world_to_image(x, y, range_m)
                        bbox_width_pixel = self._world_length_to_pixel(bbox_width, range_m)
                        bbox_length_pixel = self._world_length_to_pixel(bbox_length, range_m)
                        self._draw_rotated_box(ego_agent_channel, x_img, y_img, bbox_width_pixel, bbox_length_pixel, yaw - self.ego_yaw)
                        context_actions.append([x_rot, y_rot, 0., yaw - self.ego_yaw])
                    else:
                        trajectory = trajectories[name]
                        # TODO: ここでtrajectoryの長さがegoの長さと一致しない場合はスキップ
                        if len(trajectory) != trajectory_length:
                            continue
                        x, y, yaw, bbox_length, bbox_width, bbox_height = trajectory[t][1], trajectory[t][2], trajectory[t][4], trajectory[t][5], trajectory[t][6], trajectory[t][7]
                        x, y = x + self.ego_x, y + self.ego_y
                        x_img, y_img = self._world_to_image(x, y, range_m)
                        bbox_width_pixel = self._world_length_to_pixel(bbox_width, range_m)
                        bbox_length_pixel = self._world_length_to_pixel(bbox_length, range_m)
                        self._draw_rotated_box(other_agent_channel, x_img, y_img, bbox_width_pixel, bbox_length_pixel, yaw - self.ego_yaw)
                other_agent_channels.append(other_agent_channel)
                ego_agent_channels.append(ego_agent_channel)
            else:
                trajectory = trajectories["ego"]
                x, y, yaw, bbox_length, bbox_width, bbox_height = trajectory[t][1], trajectory[t][2], trajectory[t][4], trajectory[t][5], trajectory[t][6], trajectory[t][7]
                # x, yをegoのyawの向きに修正
                cos_yaw = np.cos(-self.ego_yaw)
                sin_yaw = np.sin(-self.ego_yaw)
                x_rot = cos_yaw * x - sin_yaw * y
                y_rot = sin_yaw * x + cos_yaw * y
                yaw = yaw - self.ego_yaw
                trajectory_labels.append([x_rot, y_rot, 0., yaw])

        for name in trajectories.keys():

            trajectory = trajectories[name]

            if len(trajectory) != trajectory_length:
                continue

            if name == "ego":
                x, y, yaw, bbox_length, bbox_width, bbox_height = trajectory[before_keyframe][1], trajectory[before_keyframe][2], trajectory[before_keyframe][4], trajectory[before_keyframe][5], trajectory[before_keyframe][6], trajectory[before_keyframe][7]
                x, y = x + self.ego_x, y + self.ego_y
                x_img, y_img = self._world_to_image(x, y, range_m, width=800, height=800)
                bbox_width_pixel = self._world_length_to_pixel(bbox_width, range_m, width=800)
                bbox_length_pixel = self._world_length_to_pixel(bbox_length, range_m, width=800)

                # Draw Bounding Box:
                rotated_corners = self._return_rotated_box(x_img, y_img, bbox_width_pixel, bbox_length_pixel, yaw - self.ego_yaw)
                x_coords, y_coords = zip(*rotated_corners)
                ax.plot(x_coords, y_coords, color='red', linestyle='-', linewidth=2)
            else:
                x, y, yaw, bbox_length, bbox_width, bbox_height = trajectory[before_keyframe][1], trajectory[before_keyframe][2], trajectory[before_keyframe][4], trajectory[before_keyframe][5], trajectory[before_keyframe][6], trajectory[before_keyframe][7]
                x, y = x + self.ego_x, y + self.ego_y
                x_img, y_img = self._world_to_image(x, y, range_m, width=800, height=800)
                bbox_width_pixel = self._world_length_to_pixel(bbox_width, range_m, width=800)
                bbox_length_pixel = self._world_length_to_pixel(bbox_length, range_m, width=800)
                rotated_corners = self._return_rotated_box(x_img, y_img, bbox_width_pixel, bbox_length_pixel, yaw - self.ego_yaw)
                x_coords, y_coords = zip(*rotated_corners)
                ax.plot(x_coords, y_coords, color='blue', linestyle='-', linewidth=2)

        # print("context_actions: ", context_actions)
        # print("-"*100)
        # print("trajectory_labels: ", trajectory_labels)
        other_agent_channels = np.stack(other_agent_channels, axis=0)
        ego_agent_channels = np.stack(ego_agent_channels, axis=0)
        rasterized_map = np.concatenate([other_agent_channels, ego_agent_channels], axis=0)
        # return rasterized_map
        return other_agent_channels, ego_agent_channels, context_actions, trajectory_labels, (self.ego_x, self.ego_y, self.ego_yaw, self.ego_bbox_length, self.ego_bbox_width, self.ego_bbox_height), ax
    
    def _draw_rotated_box(self, img, center_x, center_y, width, length, yaw):
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

    def _return_rotated_box(self, center_x, center_y, width, length, yaw):
        """
        回転した長方形を返す関数
        
        Parameters:
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
        return rotated_corners


        







