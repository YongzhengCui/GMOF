#!/bin/bash
# Install PyTorch and Python Packages

# conda create -n gmof python=3.8 -y
# conda activate gmof

pip install --upgrade pip
pip install --ignore-installed six
pip install sacred numpy scipy gym==0.11.0 matplotlib seaborn pyyaml pygame pytest probscale imageio snakeviz tensorboard-logger wandb
pip install "protobuf<3.21" -U git+https://github.com/oxwhirl/smacv2.git
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
