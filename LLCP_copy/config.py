from __future__ import division
from __future__ import print_function

import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.gpu_id = 0
__C.num_workers = 4
__C.multi_gpus = False
__C.seed = 666
# training options
__C.train = edict()
__C.train.restore = False
__C.train.lr = 0.0001
__C.train.batch_size = 32
__C.train.max_epochs = 25
#__C.train.vision_dim = 2048
__C.train.vision_dim = 512
# __C.train.train_num = 0 # Default 0 for full train set
__C.train.restore = False
__C.train = dict(__C.train)

# validation
__C.val = edict()
__C.val.flag = True
__C.val.val_num = 0 # Default 0 for full val set
__C.val = dict(__C.val)

# test
__C.test = edict()
__C.test.rate = 5.0 
__C.test.thr = 0.05
__C.test = dict(__C.test)

# model
__C.model = edict()
__C.model.latent_layer_size = 256
__C.model.latent_size = 10
__C.model.con_latent_size = 16 
__C.model = dict(__C.model)

# dataset options
__C.dataset = edict()
__C.dataset.name = ''
__C.dataset.test_object_feat= ''
__C.dataset.train_object_feat= ''
__C.dataset.appearance_feat = ''
__C.dataset.test_question_pt = ''
__C.dataset.train_question_pt = ''
__C.dataset.video_list = ''
__C.dataset.save_dir = ''
__C.dataset = dict(__C.dataset)

# experiment name
__C.exp_name = 'defaultExp'

def merge_cfg(yaml_cfg, cfg):
    if type(yaml_cfg) is not edict:
        return

    for k, v in yaml_cfg.items():
        if not k in cfg:
            raise KeyError('{} is not a valid config key'.format(k))

        old_type = type(cfg[k])
        if old_type is not type(v):
            if isinstance(cfg[k], np.ndarray):
                v = np.array(v, dtype=cfg[k].dtype)
            elif isinstance(cfg[k], list):
                v = v.split(",")
                v = [int(_v) for _v in v]
            elif cfg[k] is None:
                if v == "None":
                    continue
                else:
                    v = v
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(cfg[k]),
                                                               type(v), k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                merge_cfg(yaml_cfg[k], cfg[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            cfg[k] = v



def cfg_from_file(file_name):
    import yaml
    with open(file_name, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.Loader) )
    merge_cfg(yaml_cfg, __C)
