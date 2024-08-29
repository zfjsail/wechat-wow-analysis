# wechat-wow-analysis

Understanding WeChat User Preferences and “Wow” Diffusion.<br>
Fanjin Zhang, Jie Tang, Xueyi Liu, Zhenyu Hou, Yuxiao Dong, Jing Zhang, Xiao Liu, Ruobing Xie, Kai Zhuang, Xu Zhang, Leyu Lin, and Philip S. Yu.<br>
TKDE 2021 (accepted)

## Prerequisites

- Linux or macOS
- Python 3
- NVIDIA GPU + CUDA cuDNN

## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/zfjsail/wechat-wow-analysis.git
cd wechat-wow-analysis
```

Please install dependencies by

```bash
pip install -r requirements.txt
```
### Dataset

Due to data privacy issue in WeChat, we provide a public Weibo dataset to evaluate our prediction model.
The Weibo dataset can be downloaded from [Baidu Pan](https://pan.baidu.com/s/17SsSPJuaYMcKittbGq5qfQ?pwd=yp3m).
Unzip the file and put the _weibo_ directory into _DATA\_DIR_ specified in _settings.py_.

## How to run
```bash
cd $project_path
export CUDA_VISIBLE_DEVICES='?'  # specify which GPU(s) to be used
python train.py
```

## Cite

Please cite our paper if you use this code in your own work:

```
@article{zhang2021understanding,
  title={Understanding WeChat User Preferences and “Wow” Diffusion.},
  author={Zhang, Fanjin and Tang, Jie and Liu, Xueyi and Hou, Zhenyu and Dong, Yuxiao and Zhang, Jing and Liu, Xiao and Xie, Ruobing and Zhuang, Kai and Zhang, Xu and Lin, Leyu and Yu, Philip.},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2021}
}
```
