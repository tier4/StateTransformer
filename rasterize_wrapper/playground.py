import lanelet2
from lanelet2.core import Point2d, BoundingBox2d, BasicPoint2d, LineString3d
from lanelet2.io import load, Origin
from lanelet2.projection import UtmProjector
import numpy as np
import cv2

import os

def save_channels_as_png(rasterized_map, output_dir):
    """
    各チャンネルをPNGファイルとして保存する関数

    Parameters:
    rasterized_map (numpy.ndarray): ラスタライズされた地図データ (n_channels, width, height)
    output_dir (str): PNGファイルを保存するディレクトリ
    """
    # ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    # チャンネル名リスト（必要に応じて変更してください）
    channel_names = [
        "ego_road",
        "other_road",
        "traffic_sign",
        "traffic_light",
        "regulatory_element",
        "right_of_way",
    ]

    # 各チャンネルを保存
    for i, channel_name in enumerate(channel_names):
        if i >= rasterized_map.shape[0]:
            print(f"Warning: チャンネル数と名前リストが一致していません。チャンネル {i} はスキップされます。")
            continue
        channel_image = rasterized_map[i]
        output_path = os.path.join(output_dir, f"{channel_name}.png")
        cv2.imwrite(output_path, channel_image)
        print(f"チャンネル '{channel_name}' を保存しました: {output_path}")

def rasterize_map(osm_file, ego_x, ego_y, distance, resolution):
    # OSMファイルをロード
    projector = UtmProjector(Origin(35.0, 139.0))  # 適切な原点を設定してください
    lanelet_map = load(osm_file, projector)

    # Egoエージェントの位置を中心としたバウンディングボックスを定義
    ego_position = BasicPoint2d(ego_x, ego_y)
    min_point = BasicPoint2d(ego_x - distance, ego_y - distance)
    max_point = BasicPoint2d(ego_x + distance, ego_y + distance)
    bounding_box = BoundingBox2d(min_point, max_point)

    # バウンディングボックス内のレーンレットを取得
    #laneletlayers = lanelet_map.laneletLayer.search(bounding_box)
    laneletlayers = lanelet_map.laneletLayer
    arealayers = lanelet_map.areaLayer
    linestringlayers = lanelet_map.lineStringLayer
    polygonlayers = lanelet_map.polygonLayer
    pointlayers = lanelet_map.pointLayer
    regulatoryelementslayers = lanelet_map.regulatoryElementLayer


    # 出力画像のサイズを計算
    width = int(2 * distance / resolution)
    height = int(2 * distance / resolution)

    # 各チャンネルの初期化
    ego_road_channel = np.zeros((height, width), dtype=np.uint8)
    other_road_channel = np.zeros((height, width), dtype=np.uint8)
    traffic_sign_channel = np.zeros((height, width), dtype=np.uint8)
    traffic_light_channel = np.zeros((height, width), dtype=np.uint8)
    regulatory_element_channel = np.zeros((height, width), dtype=np.uint8)
    right_of_way_channel = np.zeros((height, width), dtype=np.uint8)

    # 座標変換のヘルパー関数
    def world_to_image(point):
        x_img = int((point.x - ego_x) / resolution)
        y_img = int((point.y - ego_y) / resolution)
        #print(point.x - ego_x, point.y - ego_y)
        return x_img + int(height / 2), int(height / 2) - y_img  # 画像座標系はy軸が下向きのため

    ego_point_2d = lanelet2.core.BasicPoint2d(ego_x, ego_y)

    # 距離が最も近い lanelet を検索 (nearest(n=1) で最も近いものを返す)
    nearest_ls = lanelet2.geometry.findNearest(lanelet_map.laneletLayer, ego_point_2d, 1)
    if len(nearest_ls) == 0:
        # 見つからない場合は None
        ego_lanelet = None
    else:
        ego_lanelet = nearest_ls[0][1]  # (distance, lanelet) のタプルが返るため [1] を取る

    # レーンレットの描画
    print(f"len(laneletlayers): {len(laneletlayers)}")
    for lanelet in laneletlayers:
        left_bound = [world_to_image(pt) for pt in lanelet.leftBound]
        right_bound = [world_to_image(pt) for pt in lanelet.rightBound]
        #print(lanelet.allWayStop, lanelet.centerline, lanelet.rightOfWay, lanelet.speedLimits, lanelet.trafficLights, lanelet.trafficSigns)

        # Egoエージェントが現在いる道路の判定（例として属性 'ego_road' を使用）
        if lanelet == ego_lanelet:
            cv2.polylines(ego_road_channel, [np.array(left_bound)], isClosed=False, color=255, thickness=2)
            cv2.polylines(ego_road_channel, [np.array(right_bound)], isClosed=False, color=255, thickness=2)
        else:
            cv2.polylines(other_road_channel, [np.array(left_bound)], isClosed=False, color=255, thickness=2)
            cv2.polylines(other_road_channel, [np.array(right_bound)], isClosed=False, color=255, thickness=2)

        # 中心線の描画（オプション）
        if lanelet.centerline is not None:
            centerline = [world_to_image(pt) for pt in lanelet.centerline]
            cv2.polylines(other_road_channel, [np.array(centerline)], isClosed=False, color=255, thickness=1)

    #areaLayer
    print(f"len(arealayers): {len(arealayers)}")
    for area in arealayers:
        pass


    #polygonLayer
    print(f"len(polygonlayers): {len(polygonlayers)}")
    for polygon in polygonlayers:
        x_imgs, y_imgs = [], []
        #print(polygon.attributes)
        for pt in polygon:
            x, y = float(pt.attributes['local_x']), float(pt.attributes['local_y'])
            dpt = lanelet2.core.BasicPoint2d(x, y)
            x_img, y_img = world_to_image(dpt)
            x_imgs.append(x_img)
            y_imgs.append(y_img)
        cv2.polylines(right_of_way_channel, [np.array([x_imgs, y_imgs]).T], isClosed=True, color=255, thickness=2)

    #regulatoryElementLayer
    print(f"len(regulatoryelementslayers): {len(regulatoryelementslayers)}")
    for regulatoryelement in regulatoryelementslayers:
        attrs = regulatoryelement.attributes
        x_imgs, y_imgs = [], []
        for v in regulatoryelement.parameters.values():
            for f in v:
                    types = f.attributes['type']
                    print(f.attributes)
                    if isinstance(f, lanelet2.core.ConstLanelet):
                        left_bound = [world_to_image(pt) for pt in f.leftBound]
                        right_bound = [world_to_image(pt) for pt in f.rightBound]
                        cv2.polylines(regulatory_element_channel, [np.array(left_bound)], isClosed=False, color=255, thickness=2)
                        cv2.polylines(regulatory_element_channel, [np.array(right_bound)], isClosed=False, color=255, thickness=2)
                    elif isinstance(f, lanelet2.core.ConstLineString3d):
                        for pt in f:
                            x, y = float(pt.attributes['local_x']), float(pt.attributes['local_y'])
                            dpt = lanelet2.core.BasicPoint2d(x, y)
                            cv2.circle(regulatory_element_channel, world_to_image(dpt), radius=50, color=255, thickness=-1)
                            x_img, y_img = world_to_image(dpt)
                            x_imgs.append(x_img)
                            y_imgs.append(y_img)
                    # g = f[0]
                    # x, y = float(g.attributes['local_x']), float(g.attributes['local_y'])
                    # dpt = lanelet2.core.BasicPoint2d(x, y)
                    # x_img, y_img = world_to_image(dpt)
                    # x_imgs.append(x_img)
                    # y_imgs.append(y_img)
        cv2.polylines(regulatory_element_channel, [np.array([x_imgs, y_imgs]).T], isClosed=False, color=255, thickness=10)












        # # 規制要素の処理
        # for reg_elem in lanelet.regulatoryElements:
        #     # 交通標識の処理
        #     if isinstance(reg_elem, lanelet2.core.TrafficSign):
        #         for sign in reg_elem.trafficSigns():
        #             #polyline
        #             x_imgs, y_imgs = [], []
        #             for pt in sign:
        #                 x_img, y_img = world_to_image(pt)
        #                 x_imgs.append(x_img)
        #                 y_imgs.append(y_img)
        #             cv2.polylines(traffic_sign_channel, [np.array([x_imgs, y_imgs]).T], isClosed=False, color=255, thickness=100)
        #     # 交通信号の処理
        #     elif isinstance(reg_elem, lanelet2.core.TrafficLight):
        #         prev_x, prev_y = None, None
        #         for light in reg_elem.trafficLights:
        #             x_imgs, y_imgs = [], []
        #             for pt in light:
        #                 x_img, y_img = world_to_image(pt)
        #                 x_imgs.append(x_img)
        #                 y_imgs.append(y_img)
        #             cv2.polylines(traffic_light_channel, [np.array([x_imgs, y_imgs]).T], isClosed=False, color=255, thickness=100)
            # # その他のオブジェクトの処理
            # elif isinstance(reg_elem, lanelet2.core.RegulatoryElement):
            #     prev_x, prev_y = None, None
            #     print(dir(reg_elem.parameters))
            #     for pt in getattr(reg_elem.parameters, "refers", []):
            #         x_img, y_img = world_to_image(pt)
            #         cv2.circle(regulatory_element_channel, (x_img, y_img), radius=50, color=255, thickness=-1)
            #         if prev_x is not None:
            #             cv2.line(regulatory_element_channel, (prev_x, prev_y), (x_img, y_img), color=255, thickness=2)
            #         prev_x, prev_y = x_img, y_img

            # elif isinstance(reg_elem, lanelet2.core.RightOfWay):
            #     for obj in reg_elem.children:
            #         if isinstance(obj, lanelet2.core.LineMarking):
            #             prev_x, prev_y = None, None
            #             for pt in obj.lineString:
            #                 x_img, y_img = world_to_image(pt)
            #                 cv2.circle(right_of_way_channel, (x_img, y_img), radius=50, color=255, thickness=-1)
            #                 if prev_x is not None:
            #                     cv2.line(right_of_way_channel, (prev_x, prev_y), (x_img, y_img), color=255, thickness=2)
            #                 prev_x, prev_y = x_img, y_img

    # チャンネルをスタックして出力
    rasterized_map = np.stack([ego_road_channel, other_road_channel, traffic_sign_channel, traffic_light_channel, regulatory_element_channel, right_of_way_channel], axis=0)
    return rasterized_map


if __name__ == '__main__':
    osm_file = "/home/kai.yamashita/Downloads/Odaiba_Map/lanelet2_map.osm"

    ego_x, ego_y = 71189.87418541731, 67461.40287274867

    raster_data = rasterize_map(osm_file, ego_x, ego_y, distance=500, resolution=0.2)
    print("raster_data.shape =", raster_data.shape)
    print(f"max: {np.max(raster_data)}, min: {np.min(raster_data)}")

    output_dir = "output_channels"
    save_channels_as_png(raster_data, output_dir)