git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
cd stylegan2-ada-pytorch
git apply ../scripts/myModification.patch
cd ..
mkdir models
# wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqv2-512x512.pkl -O models/stylegan2-afhqv2-512x512.pkl
python3 ./scripts/get_dataset_and_model.py
