from download_file import get_remote_file
import os
import zipfile

os.makedirs('./datasets', exist_ok=True)

data_url = {
    './datasets/w_afhqv2.zip': "https://drive.usercontent.google.com/download?id=1cKGmxtsOCK7jxT2haPdr24EL2RtemNDX&export=download&authuser=0&confirm=t&uuid=6f00e170-f897-433d-a879-d265c9ca0fde&at=AN_67v1Ud77qIX4Zs5iqZTt4sEfx%3A1729517969578",
    './models/stylegan2-afhqv2-512x512.pkl': "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqv2-512x512.pkl"
}

for output_path, url in data_url.items():
    get_remote_file(output_path, url)
    ## unzip in if end with zip
    if output_path.endswith('.zip'):
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(output_path.replace('.zip', ''))
        # os.remove(output_path)
        print(f"Unzipped {output_path}")