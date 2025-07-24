# Autoware Dir

## 概要
STR2をTierⅣ形式のデータで学習および推論するためのディレクトリ

### ディレクトリ構成

```
autoware/
├── rasterize_wrapper/
├── online_inference.py
├── inference.py
├── train.sh
├── train.py
```

### 各ファイルの説明

- `rasterize_wrapper/`: ラスター化のためのモジュール
- `online_inference.py`: オンラインでの推論コード(rosbagからのデータを読み込んでラスタライズ形式での描画およびmatplotlibを用いた可視化両方の実装が入っています, 現バージョン)
- `inference.py`: 推論用のスクリプト(データセットからのデータを読み込んで推論を行うものです，旧バージョン)
- `train.sh`: 学習用のコード
- `train.py`: 学習用のスクリプト
- `requirements.txt`: 依存パッケージのリスト

### 使い方

#### 方法1: Docker を使用した環境構築（推奨）

- **Dockerイメージのビルド**
    ```bash
    docker build -t str2-autoware .
    ```

- **コンテナの実行**
    ```bash
    # 開発用（インタラクティブ）
    docker run --gpus all -it -v $(pwd)/..:/workspace/STR2 str2-autoware
    ```

**Docker使用の要件:**
- NVIDIA Docker Runtime
- CUDA対応GPU
- Docker 19.03以降

#### 方法2: 仮想環境のセットアップ（手動）

- **仮想環境の作成とパッケージインストール**
    ```bash
    uv venv -p 3.12
    source .venv/bin/activate
    uv pip install -r requirements.txt
    uv pip install -e ../
    ```

- データセットの作成

    rasterize_wrapper/のREADME.mdを参照してください

- 学習
    ```bash
    bash train.sh
    ```

- 推論の可視化
    ```bash
    python online_inference.py
    ```

### ToDo
- [ ] ラスタライズ処理の改善: 現状のラスタライズ処理はLanelet2のデフォルトのPython APIを使用しており，ラスタライズしているものはLane情報くらいで，他のマップ内のオブジェクトや信号機などについてはラスタライズできていない
- [ ] 高速化: TensorRTやONNXを使用した高速化
- [ ] 大規模なモデルでの学習，推論
- [ ] Context Lengthの変更：現状はSTR2のデフォルト実装通り，Context Lengthを4にしているが，変更版の実装
- [ ] Prediction Lengthの変更：現状はSTR2のデフォルト実装通り，Prediction Lengthを80にしているが，変更版の実装
