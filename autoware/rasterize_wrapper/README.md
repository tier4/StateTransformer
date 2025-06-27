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






