python3 -m venv env
source env/bin/activate
pip install --upgrade pip
# Feel free to modify the cuda version
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt