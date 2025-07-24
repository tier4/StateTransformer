# CUDA 12.4対応のベースイメージを使用
FROM nvidia/cuda:12.4-devel-ubuntu22.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# システムパッケージの更新とPython関連の依存関係をインストール
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libgeos-dev \
    libproj-dev \
    libgdal-dev \
    libspatialindex-dev \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pythonのシンボリックリンクを作成
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# pipのアップグレード
RUN python -m pip install --upgrade pip setuptools wheel

# 作業ディレクトリの設定
WORKDIR /workspace

# requirements.txtをコピー
COPY requirements.txt .

# Pythonパッケージのインストール
# GitベースのパッケージがあるためGitが必要
RUN pip install --no-cache-dir -r requirements.txt

# STR2プロジェクト全体をコピー（setup.pyとtransformer4planningパッケージのため）
COPY ../ /workspace/STR2/

# transformer4planningパッケージのインストール
# CUDAExtensionを使用するため、CUDA環境でのビルドが必要
WORKDIR /workspace/STR2
RUN pip install -e .

# 作業ディレクトリを戻す
WORKDIR /workspace

# Jupyter Labのポート露出
EXPOSE 8888

# デフォルトコマンド
CMD ["bash"]

# オプション: Jupyter Labを自動起動したい場合は以下のコメントを外してください
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"] 