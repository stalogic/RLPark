conda create -n rlpark python=3.10
conda activate rlpark

pip install numpy==1.26

pip install torch torchvision torchaudio
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install typer
pip install wandb
pip install tqdm
pip install black
pip install opencv-python
# pip install "gym[classic_control]"
# pip install "gym[atari, accept-rom-license]"
pip install "gymnasium[classic_control]"
pip install "gymnasium[atari, accept-rom-license]"