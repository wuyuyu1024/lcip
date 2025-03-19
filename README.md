<p align="center">

  <h1 align="center">How to make dogs smile: Controlling inverse
projections by maneuvering the lost information</h1>
  <p align="center">
    <a href="https://yuwang-vis.github.io/"><strong>Yu Wang</strong></a>
    ·
    <a href="https://frederikdennig.com/"><strong>Frederik L. Dennig</strong></a>
    ·
    <a href="https://mbehrisch.github.io/"><strong>Michael Behrisch</strong></a>
    ·
    <a href="https://webspace.science.uu.nl/~telea001/"><strong>Alexandru Telea</strong></a>


  </p>
  <h2 align="center"> Submitted to <em>IEEE TVCG</em></h2>
  <div align="center">
    <img src="LCIP.gif", width="750">
  </div>
<!-- 
  <p align="center">
  <br>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href="https://twitter.com/XingangP"><img alt='Twitter' src="https://img.shields.io/twitter/follow/XingangP?label=%40XingangP"></a>
    <a href="https://arxiv.org/abs/2305.10973">
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
    <a href='https://vcai.mpi-inf.mpg.de/projects/DragGAN/'>
      <img src='https://img.shields.io/badge/DragGAN-Page-orange?style=for-the-badge&logo=Google%20chrome&logoColor=white&labelColor=D35400' alt='Project Page'></a>
    <a href="https://colab.research.google.com/drive/1mey-IXPwQC_qSthI5hO-LTX7QL4ivtPh?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  </p>
</p> -->

<!-- 
## Web Demos

[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/XingangPan/DragGAN)

<p align="left">
  <a href="https://huggingface.co/spaces/radames/DragGan"><img alt="Huggingface" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DragGAN-orange"></a>
</p> -->

## Requirements

Make sure you have CUDA graphic card, with CUDA version >= 12.1.
<!-- please follow the requirements of [NVlabs/stylegan3](https://github.com/NVlabs/stylegan3#requirements).   -->
The code was tested with Python >= 3.10, < 3.12.
The installation steps involve the following commands:

<ol>
<li>Create a new python virtual environment and activate it.

<li>Install the requirements:

```
pip install -r requirements.txt
```

<li>Then run the script to clone stylanGAN2 repo, download the pre-trained models, and download the datasets.

For Linux, run:


```sh
sh get_data_model.sh
```
If you are using windows, you can run:

``` 
.\get_data_model.bat
```

<li> TODO: download the pre-trained LCIP model with AFHQv2 dataset used in the paper. (Optional)
</ol>

<!-- ## Run Gradio visualizer in Docker 

Provided docker image is based on NGC PyTorch repository. To quickly try out visualizer in Docker, run the following:  

```sh
# before you build the docker container, make sure you have cloned this repo, and downloaded the pretrained model by `python scripts/download_model.py`.
docker build . -t draggan:latest  
docker run -p 7860:7860 -v "$PWD":/workspace/src -it draggan:latest bash
# (Use GPU)if you want to utilize your Nvidia gpu to accelerate in docker, please add command tag `--gpus all`, like:
#   docker run --gpus all  -p 7860:7860 -v "$PWD":/workspace/src -it draggan:latest bash

cd src && python visualizer_drag_gradio.py --listen
```
Now you can open a shared link from Gradio (printed in the terminal console).   
Beware the Docker image takes about 25GB of disk space!

## Download pre-trained StyleGAN2 weights

To download pre-trained weights, simply run:

```
python scripts/download_model.py
```
If you want to try StyleGAN-Human and the Landscapes HQ (LHQ) dataset, please download weights from these links: [StyleGAN-Human](https://drive.google.com/file/d/1dlFEHbu-WzQWJl7nBBZYcTyo000H9hVm/view?usp=sharing), [LHQ](https://drive.google.com/file/d/16twEf0T9QINAEoMsWefoWiyhcTd-aiWc/view?usp=sharing), and put them under `./checkpoints`.

Feel free to try other pretrained StyleGAN. -->

## Run the GUI

### Basic

To start the GUI with AFHQv2 dataset demo, run:
```
python run.py
```

To start the GUI with MNIST dataset demo, run:
```
python run.py -d mnist
```

This GUI is designed for our proposed controllable inverse projection method. You can control the inverse projection locally by dragging the sliders, and see the changes in real-time.

### More functions
This tool can also be used for general inverse projection method (e.g., [NNinv](https://webspace.science.uu.nl/~telea001/uploads/PAPERS/EuroVA19/paper.pdf), [iLAMP](https://ieeexplore.ieee.org/document/6400489), [RBF](https://www.sciencedirect.com/science/article/pii/S0097849315000230), [iMDS](http://webspace.science.uu.nl/~telea001/uploads/PAPERS/EuroVA24/paper.pdf)). However, users can only interact with the static inverse projection.


<strong>Example</strong>: To start the GUI with `Fashion-MNIST` dataset with `UMAP ` as the projection and `NNinv` as the inverse projection method, run

```
python run.py -d fashionmnist -p umap -i nninv
```

<strong>Decision map example</strong>: To start the GUI with decision map on MNIST dataset, run:
```
python run.py -d mnist -c
```

Run `python run.py -h` for the detailed instructions.


## Load the saved model used in the paper

To reproduce the style transfer application on $w$ of AFHQv2 dataset, run
```
python run.py -l
```

Then, you can load the saved $z$ by click `load z` button in the top right corner of the GUI. Then select the file in directory:
`./models/wAFHQv2_paper/z_paper_saved.npy`.

Next you could click `Map content` drop box and select 'Distance to the initial surface' to see the difference. 

<!-- You can run DragGAN Gradio demo as well, this is universal for both windows and linux:
```sh
python visualizer_drag_gradio.py
``` -->

## Acknowledgement

The latent codes of AfHQv2 dataset are obtained inverting the StyleGAN2 model using code modified from 
[NVlabs/stylegan2-ada-pytorch/projector.py](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/projector.py).
The demo on AFHQv2 dataset uses [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch) code and its pretrained model to generate images from the latent codes. 

## License
<!-- 
The code related to the DragGAN algorithm is licensed under [CC-BY-NC](https://creativecommons.org/licenses/by-nc/4.0/).
However, most of this project are available under a separate license terms: all codes used or modified from [StyleGAN3](https://github.com/NVlabs/stylegan3) is under the [Nvidia Source Code License](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt). -->

<!-- Any form of use and derivative of this code must preserve the watermarking functionality showing "AI Generated". -->

## BibTeX

```bibtex
@misc{softwareLCIP,
	title = {{LCIP} implementation source code},
	url = {https://github.com/wuyuyu1024/lcip/},
	author = {Wang, Yu and Dennig, Frederik and Behrisch, Michael and Telea, Alexandru},
	year = {2025},
}
```
