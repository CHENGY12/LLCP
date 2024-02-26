## How to Run LLCP Simulation

### Install Environment
First, please install the recent version of Pytorch and Torchvision as `pip install torch torchvision`. Then, you can install other package by running `pip install -r requirements.txt`


### Simulation Data

Please download our simulation data in the [link_function](https://drive.google.com/drive/folders/1DkSJ3JT2PdZD7TlrDc6b1uDNkakHqo1H?usp=share_link) and [link_structure](https://drive.google.com/drive/folders/1fzkVjQ0vMATajRiyiGMz6Z_MVWmz7DdD?usp=share_link) to reproduce the published results. Then, please decompress the floders as `./function_change/dataset/` and `./structure_change/dataset/` and replace the original floders as the downloaded ones.

The directory structure should look like
```
LLCP_simulation/
|–– function_change/
|   |–– dataset
|      |–– train.pkl
|      |–– test.pkl
|   |–– main.py
|   |–– models_cvae.py
|   |–– synthetic_simulate_data.py
|   |–– utils.py
|–– structure_change/
|   |–– dataset
|      |–– train.pkl
|      |–– test.pkl
|   |–– main.py
|   |–– models_cvae.py
|   |–– synthetic_simulate_data.py
|   |–– utils.py
```

If you would like to generate simulation data by yourself, please follow the below instructions.

For function change setting
```
cd function_change
python synthetic_simulate_data.py
```

For structure change setting
```
cd structure_change
python synthetic_simulate_data.py
```


### Run Scripts


Please run the following code to to reproduce the experiment results.

For function change setting
```
cd function_change
python main.py
```

For structure change setting
```
cd structure_change
python main.py
```