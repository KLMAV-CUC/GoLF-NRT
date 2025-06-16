## Data preparation

Please follow original setting of [GNT](https://github.com/VITA-Group/GNT) to prepare the data.

## Usage (under construction)

Please prepare the environment as follows.

```
conda create -n golfnrt python=3.8
conda activate golfnrt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Additionally dependencies include:
```
ConfigArgParse
imageio
matplotlib
opencv_contrib_python
Pillow
scipy
imageio-ffmpeg
lpips
scikit-image
termcolor
easydict
scikit-video
timm
einops
```

After setting up the environment and perparing the data, you can train and test the code following [GNT](https://github.com/VITA-Group/GNT).

## Evaluation
```
python eval.py --config configs/golf_full.txt --expname golf_full --chunk_size 4096 --run_val
```


## Citing
If you find our work helpful, please feel free to use the following BibTex entry:
```BibTeX
@article{zhu2023caesarnerf,
    author  = {Wang, You and Fang, Li and Zhu, Hao and Hu, Fei and Ye, Long and Ma, Zhan},
    title   = {GoLF-NRT: Integrating Global Context and Local Geometry for Few Shot View Synthesis},
    booktitle = {CVPR},
    year    = {2025}
}
```
