## How to Run LLCP

### Install Environment
First, please install the recent version of Pytorch and Torchvision as `pip install torch torchvision`. Then, you can install other package by running `pip install -r requirements.txt`


### Download Data
Due the time consuming data pre-process (tracking the variable and obtain the CLIP feature), we provide the processed features used in our experiments with an an Anonymous link. Please download the data and model in this [link1](https://drive.google.com/drive/folders/17TDv6CxenKlyr8W2gnmrojnGP82kwlqp?usp=share_link) and this [link2](https://drive.google.com/drive/folders/1BGBiY1_qp0ElHORLi4y0AEyh79Hnn9oN?usp=share_link). Then please decompress the floders as `./data/` and `./results/` and replace the original floders as the downloaded ones.

The directory structure should look like
```
LLCP/
|–– config.py
|–– configs/
|–– data/
|   |–– object_test_feat/
|   |–– appearance_feat_rn50.h5
|   |–– test_questions.pt
|–– DataLoader.py
|–– models_cvae.py
|–– requirements.txt
|–– results/
|   |–– .../model_cvae49.pt
|–– Test_LLCP.md
|–– validate_causal.py
```


### Run Scripts

The running scripts in `scripts/`. You can run the commands `python validate_causal.py` under `LLCP`.