# rasterize_wrapper

## 概要

Autowareのデータセットをラスター化するためのラッパーです。

### 各ファイルの説明

- `create_dataset.py`: データセットを作成するためのスクリプト
- `rasterize_class.py`: ラスター化を行うクラス
- `rasterize_online.py`: 推論時にオンラインでラスター化を行うクラス(matplotlibの描画処理も含む，online_inference.pyで使用)

### 使い方

データセットの作成

```bash
python create_dataset.py --osm_file_path <osm_file_path> --db_list_path_file <db_list_path_file> --dataset_dir <dataset_dir>
```

osm_file_path: osmファイルのパス
db_list_path_file: rosbagのパスが書かれたファイルのパス
dataset_dir: データセットを保存するディレクトリのパス

db_list_path_fileのexample:

```
/home/acf15382lp/Downloads/rosbag-data/0b676675-21db-4b26-a1c0-04e646a57b35
/home/acf15382lp/Downloads/rosbag-data/5716498d-549d-4b10-ba65-279231e599bd
/home/acf15382lp/Downloads/rosbag-data/e053d784-19a7-4b0d-8d71-23d495221248
/home/acf15382lp/Downloads/rosbag-data/fe1f9718-7e47-422d-b849-c90f964bdfc6
/home/acf15382lp/Downloads/rosbag-data/3de1b10c-9711-4fdf-9786-73d0d0da9ce9
```

### モデルの入出力の説明

#### 入力

dictのkey:

- `high_res_raster`: 短い範囲でのEgo Agentを中心としたBEV形式のRasterize画像 (224, 224, 58)
    
    ラスタライズの各Channelの説明
    - 0~1: Ego Agentが走行しているLane
    - 2~21: それ以外のLaneを道路の種類ごとに画像化
    - 22~25: 信号機
    - 26~57: 他車両や歩行者などの他Agentの軌跡を種類ごとに画像化 Agent Type 8 x Context Length 4 = 32
- `low_res_raster`: 長い範囲でのEgo Agentを中心としたBEV形式のRasterize画像 (224, 224, 58)
- `context_actions`: 過去4ステップのEgo Agentの行動(4, 4), 行動は(x, y, z, yaw)の形式 z = 0
- `trajectory_label`: 未来80ステップのEgo Agentの軌跡のラベル(80, 4), 同様に(x, y, z, yaw)の形式 z = 0

#### 出力

dictのkey:

- `logits`: 未来80ステップのEgo Agentの軌跡と5つのKey Pointのlogits(85, 4)
- `pred_dict`: logitのdict
    - `traj_logits`: 未来80ステップのEgo Agentの軌跡のlogits(80, 4)
    - `kp_logits`: 5つのKey Pointのlogits(5, 4)
- `loss_items`: 損失のdict
- `hidden_states`: 隠れ状態





