conda create -y --name occamnets python==3.8
source activate occamnets

# Pytorch + Lightning
pip install pytorch-lightning
conda install --yes pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install -c anaconda protobuf

# Hydra
pip install hydra-core --upgrade
pip install hydra-joblib-launcher --upgrade
conda install --yes -c anaconda pyyaml
