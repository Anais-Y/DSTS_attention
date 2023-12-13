import torch
import numpy as np

loss_model = ['14_0.48', '15_0.24', '16_0.31', '17_0.28', '18_0.18',
              '20_0.25', '9_0.27', '4_0.32',]  # 有module '1_0.22',
# loss_model = ['2_0.44', '10_0.23', '12_0.44', '13_0.19','26_0.29', '29_0.21', '32_0.48', '3_0.25', '7_0.22',]
weight = ['0.A', '0.weight_cross_time', '0.weight_cross_time2',
          '1.A', '1.weight_cross_time', '1.weight_cross_time2', '2.A', '2.weight_cross_time',
          '2.weight_cross_time2', '3.A', '3.weight_cross_time', '3.weight_cross_time2']
for loss in loss_model:
    model_checkpoint = torch.load(f'./garage/DEAP/exp_{loss}_best_model.pth', map_location='cpu')
    print(model_checkpoint.keys())
    for w in weight:
        print_weight = model_checkpoint[f'module.STSGCLS.3.constructed_adj.{w}'].numpy()
        np.save(f'Weight/exp{loss[:-5]}-{w}.npy', print_weight)
