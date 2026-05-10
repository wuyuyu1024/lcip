<div align="center">
  <h1>LCIP: Loss-Controlled Inverse Projection of High-Dimensional Data</h1>
  <p align="center">
    <a href="https://yuwang-vis.github.io/"><strong>Yu Wang</strong></a>
    ·
    <a href="https://frederikdennig.com/"><strong>Frederik L. Dennig</strong></a>
    ·
    <a href="https://mbehrisch.github.io/"><strong>Michael Behrisch</strong></a>
    ·
    <a href="https://webspace.science.uu.nl/~telea001/"><strong>Alexandru Telea</strong></a>
  </p>
  <h3><a href="https://doi.org/10.48550/arXiv.2602.11141">arXiv:2602.11141</a></h3>
  <img src="LCIP.gif" width="750">
  <p align="center">
    <a href="https://www.youtube.com/watch?v=9h5rFLiETS0"><strong>Demo Video</strong></a>
  </p>
</div>

## Requirements

LCIP requires a CUDA-capable graphics card with CUDA >= 12.1. The project supports Python >= 3.10 and < 3.12.

On Linux, the Qt GUI also requires native system libraries. On Arch, install `libxkbcommon` before launching the app.
The installation steps involve the following commands:

<ol>
<li>Install <code>uv</code> and sync the project environment:

```sh
uv sync --python 3.11
```

<li>Then run the script to clone the StyleGAN2 dependency, download the pretrained models, and download the datasets.

For Linux, run:


```sh
sh get_data_model.sh
```
If you are using Windows, run:

```bat
.\get_data_model.bat
```
</ol>

## Run tests

Use `pytest` for the test suite:

```sh
uv run pytest -q
```

## Run the GUI

### Basic

To start the GUI with AFHQv2 dataset demo, run:

```sh
uv run python run.py
```

To start the GUI with MNIST dataset demo, run:

```sh
uv run python run.py -d mnist
```

This GUI is designed for our proposed controllable inverse projection method. You can control the inverse projection locally by dragging the sliders, and see the changes in real-time.

### More functions
This tool can also be used for general inverse projection methods such as [NNinv](https://webspace.science.uu.nl/~telea001/uploads/PAPERS/EuroVA19/paper.pdf), [iLAMP](https://ieeexplore.ieee.org/document/6400489), [RBF](https://www.sciencedirect.com/science/article/pii/S0097849315000230), and [iMDS](http://webspace.science.uu.nl/~telea001/uploads/PAPERS/EuroVA24/paper.pdf). In these modes, users interact with a static inverse projection.


<strong>Example</strong>: To start the GUI with `Fashion-MNIST`, `UMAP` as the projection, and `NNinv` as the inverse projection method, run:

```sh
uv run python run.py -d fashionmnist -p umap -i nninv
```

<strong>Decision map example</strong>: To start the GUI with a decision map on MNIST, run:

```sh
uv run python run.py -d mnist -c
```

Run `uv run python run.py -h` for the detailed instructions.

## Load the saved model used in the paper

To reproduce the style transfer application on $w$ of AFHQv2 dataset, run

```sh
uv run python run.py -l
```

Then load the saved $z$ by clicking the `load z` button in the top-right corner of the GUI and selecting:
`./models/wAFHQv2_paper/z_paper_saved.npy`.

Next, open the `Map content` dropdown and select `Distance to the initial surface`.


## Acknowledgement

The latent codes of AFHQv2 dataset are obtained by inverting the StyleGAN2 model using code modified from
[NVlabs/stylegan2-ada-pytorch/projector.py](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/projector.py).
The demo on AFHQv2 dataset uses [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch) code and its pretrained model to generate images from the latent codes.

## Citation

GitHub can read the repository citation metadata from [`CITATION.cff`](CITATION.cff). If you use LCIP, please cite the software and associated paper.

## License

The original LCIP source code in this repository is released under the MIT License. See [`LICENSE`](LICENSE) for details. Downloaded third-party dependencies, pretrained models, datasets, and external assets retain their own licenses and terms.

## BibTeX

```bibtex
@misc{softwareLCIP,
	title = {{LCIP} implementation source code},
	url = {https://github.com/wuyuyu1024/lcip/},
	author = {Wang, Yu and Dennig, Frederik and Behrisch, Michael and Telea, Alexandru},
	year = {2026},
}
```
