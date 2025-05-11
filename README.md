<div align="center">
<h1>测试 （CVPR 2025）IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera</h1>
</div>


## 配置测试

```bash
git clone https://github.com/ERGlab/IncEventGS.git --recursive

# rm -rf .git

conda create -n iegs python=3.10
conda activate iegs
# conda remove --name iegs --all

conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit # install CUDA toolkit
# Install the pytorch first (Please check the cuda version)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt

pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install h5py hdf5plugin scikit-learn jaxtyping kornia

pip install -U  "huggingface-hub<0.26.0"
pip install pytorch_msssim
pip install tyro
pip install diffusers==0.27.1
pip install accelerate
pip install -U "transformers<=4.48.1"


```

* 下载数据集[link](https://westlakeu-my.sharepoint.com/:f:/g/personal/cvgl_westlake_edu_cn/EooEsF6LJPJHnEqbTYo4qO0B4-dlbznGZlLy9BDLPAx_og?e=wmqeFP)
* 下载ondrive数据[blog](https://kwanwaipang.github.io/File/Blogs/Poster/ubuntu%E5%91%BD%E4%BB%A4%E8%A1%8C%E4%B8%8B%E8%BD%BD%E6%95%B0%E6%8D%AE.html)