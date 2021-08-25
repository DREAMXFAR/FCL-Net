import torch
import yaml
import argparse
import random
import numpy

from fcl_pipeline import *
from bdcn_pipeline import *
from attrdict import AttrDict

###############
# parse cfg
###############
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', dest='cfg', required=True, help='path to config file')
parser.add_argument('--mode', dest='mode', required=True, help='path to config file')
parser.add_argument('--time', dest='time', required=True, help='path to config file')
args = parser.parse_args()

cfg_file = args.cfg
print('cfg_file: ', cfg_file)

with open('config/' + cfg_file, 'r') as f:
    cfg = AttrDict(yaml.load(f))

cfg.path = cfg_file
cfg.time = args.time
print(cfg)

random_seed = cfg.TRAIN.random_seed
if random_seed > 0:
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    numpy.random.seed(random_seed)

########################################

if cfg.MODEL.mode == 'BDCN':
    pipeline = BDCNPipeline(cfg)
else:
    pipeline = FCLPipeline(cfg)

if args.mode == 'train':
    pipeline.train()
elif args.mode == 'test_ms':
    param_path = r'/path/to/parameters'
    pipeline.test_ms(param_path=param_path, mode='s')
else:
    print('Else Mode!!!')
