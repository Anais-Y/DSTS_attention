from utils import *
import argparse
from vis_model import *
import numpy as np
import configparser
import ast
import torch

# 加载保存的.pth文件
saved_state = torch.load("## .pth")

# 提取保存的模型状态和超参数
model_state = saved_state['state_dict']
hyperparameters = saved_state['hyperparams']
device = torch.device("cpu")
model = STSGCN(**hyperparameters).to(device)
model.load_state_dict(model_state)

# 现在模型已经加载并可以使用了
