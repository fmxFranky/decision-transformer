# !/bin/bash
# rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list
# apt-get update
# DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
#     apt-utils build-essential ca-certificates dpkg-dev pkg-config software-properties-common module-init-tools \
#     cifs-utils openssh-server nfs-common net-tools iputils-ping iproute2 locales htop tzdata \
#     tar wget git swig vim curl tmux zip unzip rar unrar sudo patchelf \
#     libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev libglfw3 libglew2.0 
# sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen
# locale-gen en_US.UTF-8
# dpkg-reconfigure --frontend=noninteractive locales
# update-locale LANG=en_US.UTF-8
# echo "Asia/Shanghai" > /etc/timezone
# rm -f /etc/localtime
# rm -rf /usr/share/zoneinfo/UTC
# dpkg-reconfigure --frontend=noninteractive tzdata

mkdir ~/.mujoco
wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip
unzip mujoco.zip -d ~/.mujoco
rm mujoco.zip
cp -r ~/.mujoco/mujoco200_linux/ ~/.mujoco/mujoco200/
cp ./mjkey.txt ~/.mujoco/
echo 'export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
echo 'export MUJOCO_KEY_PATH="$HOME/.mujoco$MUJOCO_KEY_PATH"' >> ~/.bashrc
source ~/.bashrc
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"
export MUJOCO_KEY_PATH="$HOME/.mujoco$MUJOCO_KEY_PATH"

conda install ruamel.yaml flake8 yapf -y
pip install --upgrade scikit-image absl-py tb-nightly pyparsing imageio-ffmpeg termcolor 
pip install --upgrade hydra-core gym kornia wandb mujoco_py transformers numpy
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl

wandb login 31ce01e4120061694da54a54ab0dafbee1262420
# cd data && python download_d4rl_datasets.py