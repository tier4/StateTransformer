import numpy as np
import lanelet2
import lanelet2.core
import lanelet2.io
import lanelet2.projection
import lanelet2.geometry
import cv2
from shapely.geometry import Point, Polygon
from lanelet2.core import TrafficLight, RegulatoryElement

from typing import Tuple

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

def rasterize_lanelet_map(
    osm_path: str,
    ego_xy: Tuple[float, float],
    max_distance: float = 50.0,
    resolution: float = 0.1,
    origin_lat: float = 0.0,
    origin_lon: float = 0.0,
) -> np.ndarray:
    """
    Lanelet2 のマップ(OSM)を読み込み, ego エージェントを中心とした一定範囲を
    rasterize (白黒画像化)して返す関数。

    Parameters
    ----------
    osm_path : str
        読み込む OSM ファイルへのパス
    ego_xy : (float, float)
        ego エージェントの (x, y) 位置 [投影座標系 or ローカル座標系上での位置]
    max_distance : float
        ego 位置を中心とした描画範囲 [メートル]
    resolution : float
        1 ピクセルあたりのメートル数
    origin_lat : float
        マップの投影用 Origin(経度)
    origin_lon : float
        マップの投影用 Origin(緯度)

    Returns
    -------
    raster : np.ndarray
        形状が (n_channels, height, width) のラスタ画像。
        例として、
          raster[0, :, :] : ego が属する道路レーン (white=1/black=0)
          raster[1, :, :] : その他の道路レーン
          raster[2, :, :] : 交通標識
          raster[3, :, :] : 信号機
        のようにチャンネルを割り当てている。

        height, width は 画像上の縦・横ピクセル数 ( = 2*max_distance / resolution 程度 )
    """
    #--------------------------------------------------------------------------
    # 1. Lanelet2 マップの読み込み
    #--------------------------------------------------------------------------
    # 投影 (UTM, LocalFrame など好きなプロジェクションを使用)
    # ここでは例として UTMProjector を使用するが、用途に応じて変更可能
    projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(origin_lat, origin_lon))
    # OSM ファイルをロードし、lanelet2.Map オブジェクトを取得
    lanelet_map = lanelet2.io.load(osm_path, projector)

    #--------------------------------------------------------------------------
    # 2. ego 位置のレーン(let)を特定する
    #--------------------------------------------------------------------------
    #   - ego_xy がどのレーンレットに属しているかを検索
    #   - 属していれば「ego エージェントがいるレーン」として扱う
    ego_point_2d = lanelet2.core.BasicPoint2d(ego_xy[0], ego_xy[1])

    # 距離が最も近い lanelet を検索 (nearest(n=1) で最も近いものを返す)
    nearest_ls = lanelet2.geometry.findNearest(lanelet_map.laneletLayer, ego_point_2d, 1)
    if len(nearest_ls) == 0:
        # 見つからない場合は None
        ego_lanelet = None
    else:
        ego_lanelet = nearest_ls[0][1]  # (distance, lanelet) のタプルが返るため [1] を取る

    #--------------------------------------------------------------------------
    # 3. ラスタ画像領域 (ピクセル座標系) の設定
    #--------------------------------------------------------------------------
    #   - ego 位置を画像の中心とし、max_distance 分だけ上下左右に広げる
    #   - 1 ピクセルあたり resolution [m/pixel] のスケールで画像を作成
    #
    #   height = 2 * max_distance / resolution
    #   width  = 2 * max_distance / resolution
    #
    #   Pixel (u, v) の実座標 (x, y) は
    #     x = ego_x + (u - width/2 ) * resolution
    #     y = ego_y + (v - height/2) * resolution
    #   のように変換する。
    #--------------------------------------------------------------------------
    half_size = max_distance
    width  = int((2.0 * half_size) / resolution)
    height = int((2.0 * half_size) / resolution)

    # チャンネル数 (例: [ego road, other roads, traffic signs, traffic signals] = 4)
    n_channels = 4
    raster = np.zeros((n_channels, height, width), dtype=np.uint8)

    #--------------------------------------------------------------------------
    # 4. 交通標識, 信号機をまとめて取得する
    #--------------------------------------------------------------------------
    # lanelet2 のマップでは、trafficSigns や trafficLights は
    # RegulatoryElements に含まれている場合が多い。
    # regulatoryElements の中から type で判定する方法などが考えられる。
    #
    # ここでは簡易的に
    #   - trafficSignsLayer (道路標識)
    #   - trafficLightsLayer (信号機)
    # があればそこから取得する例を示す。
    #--------------------------------------------------------------------------
    traffic_lights = []
    traffic_signs = []

    for lanelet in lanelet_map.laneletLayer:
        #print(len(lanelet.regulatoryElements))
        for reg_elem in lanelet.regulatoryElements:
            if isinstance(reg_elem, TrafficLight):
                traffic_lights.append(reg_elem)
            elif isinstance(reg_elem, RegulatoryElement):
                traffic_signs.append(reg_elem)

    #--------------------------------------------------------------------------
    # 5. 各 pixel について、どのチャンネルに該当するかチェックして描画
    #    (最も単純な「各ピクセルごとに判定」する方法の例)
    #
    #    高速化したい場合は、Shapely などでポリゴンを rasterize するアプローチも検討可
    #--------------------------------------------------------------------------
    # laneletLayer の全レーンレットを取得
    all_lanelets = list(lanelet_map.laneletLayer)

    # 関数: ピクセル座標(u,v) -> ワールド座標(x, y)
    def pixel_to_world(u: int, v: int) -> Tuple[float, float]:
        x = ego_xy[0] + (u - width  / 2.0) * resolution
        y = ego_xy[1] + (v - height / 2.0) * resolution
        return (x, y)

    # レーン(let)を Shapely で扱いたい場合は下記のようにポリゴン化できるが、
    # ここではピクセル単位のポリゴン内判定が重いため、最も直接的な lanelet2.geometry の
    # within() / intersects2d() 等を用いてポイントが含まれるか判定する方法を紹介する。
    #
    #   import shapely.geometry
    #   poly = shapely.geometry.Polygon([(p.x, p.y) for p in lanelet2.geometry.to2D(lanelet).basicPolygon()])
    #
    #   if poly.contains(shapely.geometry.Point(x, y)):
    #       # 中にいる
    #
    # など。

    for v in range(height):
        for u in range(width):
            # ピクセル中心座標で判定 (サンプリングする点を中心寄りにしたい場合は +0.5 など調整)
            x, y = pixel_to_world(u, v)
            point_2d = lanelet2.core.BasicPoint2d(x, y)

            #-------------------------
            # [ チャンネル 0: ego エージェントがいるレーン ]
            #-------------------------
            shapely_point = Point(x, y)
            shapely_polygon = Polygon([(p.x, p.y) for p in lanelet2.geometry.to2D(ego_lanelet.polygon2d())])
            if ego_lanelet is not None:
                if shapely_polygon.contains(shapely_point):
                    raster[0, v, u] = 1  # white

            #-------------------------
            # [ チャンネル 1: その他のレーン(let) ]
            #-------------------------
            # ego_lanelet と同じものは除きつつ、レーン内に含まれていれば描画
            for ln in all_lanelets:
                shapely_polygon = Polygon([(p.x, p.y) for p in lanelet2.geometry.to2D(ln.polygon2d())])
                if ln.id == (ego_lanelet.id if ego_lanelet else -1):
                    continue
                if shapely_polygon.contains(shapely_point):
                    raster[1, v, u] = 1
                    break  # 重複して描画しないよう1回でも含まれれば break

            #-------------------------
            # [ チャンネル 2: 交通標識 (TrafficSign) ]
            #-------------------------
            # traffic_signs は通常 point か linestring か、あるいは
            # RegulatoryElement が付加情報として持っているケースが多い。
            # ここでは単純に、trafficSigns レイヤにある要素の座標に近ければ 1 とするなど、
            # 必要に応じてロジックを実装する。
            # (実際は bounding box 判定や、traffic sign オブジェクトの幾何情報を取得して
            #  中に入っているかを判定する、などが考えられる)

            for sign in traffic_signs:
                for element in sign.parameters:
                    if isinstance(element, lanelet2.core.LineString3d):
                        element_2d = lanelet.geometry.to2D(element)
                        num_points = len(element_2d)
                        if num_points == 0:
                            continue
                        center_x = sum(pt.x for pt in element_2d) / num_points
                        center_y = sum(pt.y for pt in element_2d) / num_points
                        sign_center = lanelet2.core.BasicPoint2d(center_x, center_y)
                    elif isinstance(element, lanelet2.core.Point3d):
                        sign_center = lanelet2.geometry.to2D(element).basicPoint()
                    else:
                        continue
                    # 距離の計算
                    dx = x - sign_center.x
                    dy = y - sign_center.y
                    dist2 = dx * dx + dy * dy
                    if dist2 < (0.5 * 0.5):
                        raster[2, v, u] = 1
                        break

            #-------------------------
            # [ チャンネル 3: 信号機 (TrafficLight) ]
            #-------------------------
            for tlight in traffic_lights:
               pass

    return raster

if __name__ == "__main__":
    # サンプル実行
    osm_file = "/data/home/kai.yamashita/Downloads/Odaiba_Map/lanelet2_map.osm"
    ego_x, ego_y = (0, 0)  # ego の座標(例)
    raster_data = rasterize_lanelet_map(
        osm_file,
        (ego_x, ego_y),
        max_distance=10.0,
        resolution=1.0,
        origin_lat=35.0,  # 例
        origin_lon=139.0    # 例
    )

    print("raster_data.shape =", raster_data.shape)
    save_channels_as_png(raster_data, "output_dir")
    # 例: (4, height, width)
    # チャンネル数=4, height= (2*50)/0.2 = 500, width= 500