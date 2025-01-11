## Installation

### Set Up Environment
Clone the repository.
```
git clone https://github.com/IDGallagher/MotionVideoSearch
cd MotionVideoSearch
```

### Install dependencies

[Optional] Create a conda environment.
```
conda create -n mvs python=3.9
conda activate mvs
```

Install the [PyTorch and TorchVision](https://pytorch.org/get-started/locally/) versions that are compatible with your CUDA configuration.
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install ffmpeg 6.x
```
conda install -c conda-forge ffmpeg=6.*
```

Install other dependencies.
```
pip install tqdm matplotlib einops einshape scipy timm lmdb av mediapy typer imageio imageio-ffmpeg requests opencv-python

```
Install faiss https://github.com/facebookresearch/faiss
```
conda install -c pytorch faiss-cpu=1.9.0
```

[Optional] Download watermark removal repo
```
git clone https://github.com/l-comm/WatermarkRemoval.git
```