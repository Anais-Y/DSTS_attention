from utils import *
import argparse
from vis_model import *
import tqdm
import numpy as np
import pandas as pd
import configparser
import ast
import torch

import matplotlib.pyplot as plt

DATASET = 's01_test'  # PEMSD4 or PEMSD8

config_file = './configFiles/DEAP/{}.conf'.format(DATASET)
config = configparser.ConfigParser()
config.read(config_file)

# parser = argparse.ArgumentParser(description='arguments')
# parser.add_argument('--no_cuda', action="store_true", help="没有GPU")
# parser.add_argument('--data', type=str, default=config['data']['data'], help='data path')
# # parser.add_argument('--sensors_distance', type=str, default=config['data']['sensors_distance'], help='节点距离文件')
# # parser.add_argument('--column_wise', type=eval, default=config['data']['column_wise'],
# # help='是指列元素的级别上进行归一，否则是全样本取值')
# parser.add_argument('--normalizer', type=str, default=config['data']['normalizer'], help='归一化方式')
# parser.add_argument('--batch_size', type=int, default=config['data']['batch_size'], help="batch大小")
#
# parser.add_argument('--num_of_vertices', type=int, default=config['model']['num_of_vertices'], help='传感器数量')
# parser.add_argument('--construct_type', type=str, default=config['model']['construct_type'],
#                     help="构图方式  {connectivity, distance}")
# parser.add_argument('--in_dim', type=int, default=config['model']['in_dim'], help='输入维度')
# parser.add_argument('--hidden_dims', type=list, default=ast.literal_eval(config['model']['hidden_dims']),
#                     help='中间各STSGCL层的卷积操作维度')
# parser.add_argument('--first_layer_embedding_size', type=int, default=config['model']['first_layer_embedding_size'],
#                     help='第一层输入层的维度')
# parser.add_argument('--out_layer_dim', type=int, default=config['model']['out_layer_dim'], help='输出模块中间层维度')
# parser.add_argument("--history", type=int, default=config['model']['history'], help="每个样本输入的离散时序")
# # parser.add_argument("--horizon", type=int, default=config['model']['horizon'], help="每个样本输出的离散时序")
# parser.add_argument("--strides", type=int, default=config['model']['strides'], help="滑动窗口步长，local时空图使用几个时间步构建的，默认为3")
# parser.add_argument("--temporal_emb", type=eval, default=config['model']['temporal_emb'], help="是否使用时间嵌入向量")
# parser.add_argument("--spatial_emb", type=eval, default=config['model']['spatial_emb'], help="是否使用空间嵌入向量")
# # parser.add_argument("--use_mask", type=eval, default=config['model']['use_mask'], help="是否使用mask矩阵优化adj")
# parser.add_argument("--activation", type=str, default=config['model']['activation'], help="激活函数 {relu, GlU}")
#
# parser.add_argument('--log_file', default=config['test']['log_file'], help='log file')
# parser.add_argument('--checkpoint', type=str, help='')
#
# args = parser.parse_args()
parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--no_cuda', action="store_true", help="没有GPU")
parser.add_argument('--data', type=str, default=config['data']['data'], help='data path')
parser.add_argument('--normalizer', type=str, default=config['data']['normalizer'], help='归一化方式')
parser.add_argument('--batch_size', type=int, default=config['data']['batch_size'], help="batch大小")

parser.add_argument('--num_of_vertices', type=int, default=config['model']['num_of_vertices'], help='传感器数量')
parser.add_argument('--construct_type', type=str, default=config['model']['construct_type'],
                    help="构图方式  {connectivity, distance}")
parser.add_argument('--in_dim', type=int, default=config['model']['in_dim'], help='输入维度')
parser.add_argument('--num_layers', type=int, default=ast.literal_eval(config['model']['num_layers']),
                    help='STSGCL个数')
parser.add_argument('--first_layer_embedding_size', type=int, default=config['model']['first_layer_embedding_size'],
                    help='第一层输入层的维度')
parser.add_argument('--out_layer_dim', type=int, default=config['model']['out_layer_dim'], help='输出模块中间层维度')
parser.add_argument("--history", type=int, default=config['model']['history'], help="每个样本输入的离散时序")
parser.add_argument("--strides", type=int, default=config['model']['strides'], help="滑动窗口步长，local时空图使用几个时间步构建的，默认为3")
parser.add_argument("--num_gcn", type=int, default=config['model']['num_gcn'], help="并行卷积核数量")

parser.add_argument("--temporal_emb", type=eval, default=config['model']['temporal_emb'], help="是否使用时间嵌入向量")
parser.add_argument("--spatial_emb", type=eval, default=config['model']['spatial_emb'], help="是否使用空间嵌入向量")
parser.add_argument("--activation", type=str, default=config['model']['activation'], help="激活函数 {relu, GlU}")

parser.add_argument('--seed', type=int, default=config['train']['seed'], help='种子设置')
parser.add_argument("--learning_rate", type=float, default=config['train']['learning_rate'], help="初始学习率")
parser.add_argument("--lr_decay", type=eval, default=config['train']['lr_decay'], help="是否开启初始学习率衰减策略")
parser.add_argument("--lr_decay_step", type=str, default=config['train']['lr_decay_step'], help="在几个epoch进行初始学习率衰减")
parser.add_argument("--lr_decay_rate", type=float, default=config['train']['lr_decay_rate'], help="学习率衰减率")
parser.add_argument('--epochs', type=int, default=config['train']['epochs'], help="训练代数")
parser.add_argument('--print_every', type=int, default=config['train']['print_every'], help='几个batch报训练损失')
parser.add_argument('--save', type=str, default=config['train']['save'], help='保存路径')
parser.add_argument('--expid', type=int, default=config['train']['expid'], help='实验 id')
parser.add_argument('--max_grad_norm', type=float, default=config['train']['max_grad_norm'], help="梯度阈值")

parser.add_argument('--patience', type=int, default=config['train']['patience'], help='等待代数')
parser.add_argument('--log_file', default=config['train']['log_file'], help='log file')
parser.add_argument('--checkpoint', type=str, help='')

args = parser.parse_args()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
log = open(args.log_file, 'w')
log_string(log, str(args))


def visualize_adjacency_matrix(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name == 'STSGCLS.0.STSGCMS.1.gcn_operations.0.adj':
                adj_matrix = param.cpu().numpy()
                np.save('./010adj_matrix_s5.npy', adj_matrix)
                break
        plt.figure(figsize=(10, 10))
        plt.imshow(adj_matrix, cmap="YlGnBu")
        plt.colorbar()
        plt.title("Adjacency Matrix Visualization")
        plt.savefig('Adjacency010.pdf')


def main():
    # load data

    dataloader = load_dataset(dataset_dir=args.data,
                              normalizer=args.normalizer,
                              batch_size=args.batch_size,
                              valid_batch_size=args.batch_size,
                              test_batch_size=args.batch_size)

    scaler = dataloader['scaler']

    # model = STSGCN(
    #     history=args.history,
    #     num_of_vertices=args.num_of_vertices,
    #     in_dim=args.in_dim,
    #     hidden_dims=args.hidden_dims,
    #     first_layer_embedding_size=args.first_layer_embedding_size,
    #     out_layer_dim=args.out_layer_dim,
    #     activation=args.activation,
    #     temporal_emb=args.temporal_emb,
    #     spatial_emb=args.spatial_emb,
    #     strides=args.strides
    # ).to(device)

    model = STSGCN(
        history=args.history,
        num_of_vertices=args.num_of_vertices,
        in_dim=args.in_dim,
        num_layers=args.num_layers,
        first_layer_embedding_size=args.first_layer_embedding_size,
        out_layer_dim=args.out_layer_dim,
        activation=args.activation,
        temporal_emb=args.temporal_emb,
        spatial_emb=args.spatial_emb,
        strides=args.strides,
        num_gcn=args.num_gcn
    ).to(device)

    model.eval()

    log_string(log, '加载模型成功')

    outputs = []
    realy = torch.Tensor(dataloader['y_train']).to(device)  # 加载训练集标签

    realy = realy[..., 0]
    # [B, T, N]

    x = torch.Tensor(dataloader['x_train']).to(device)
    y = torch.Tensor(dataloader['y_train']).to(device)

    modified_tensor = y.clone()  # 先复制原tensor
    modified_tensor[:, 0] = 9 - modified_tensor[:, 0]
    sums = modified_tensor.sum(dim=1)

    # 找到和最大的索引
    _, index_of_max_sum = torch.max(sums, dim=0)

    index = index_of_max_sum.item()
    x = torch.unsqueeze(x[index, ...], dim=0)
    print(index, x.shape, y.shape)
    y = torch.unsqueeze(y[index, ...], dim=0)
    # for iter, (x, y) in tqdm.tqdm(enumerate(dataloader['train_loader'].get_iterator())):
    #     testx = torch.Tensor(x).to(device)
    #     with torch.no_grad():
    #         preds = model(testx)
    #         # [B, T, N]
    #
    #         outputs.append(preds)
    # model.load_state_dict(torch.load(args.checkpoint))  # 多GPU保存，多GPU加载(但可能会导致后面没法画图)，但GPU保存，单GPU加载

    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.checkpoint).items()})
    # 多GPU保存， 单GPU加载

    # model.eval()

    # log_string(log, '加载模型成功')

    # visualize_adjacency_matrix(model)
    with torch.no_grad():
        output = model(x).numpy()
        output = np.squeeze(output)
        print(output.shape)
        print(y)
        np.save('output_layer_dataLAHV.npy', output)

if __name__ == "__main__":
    main()
    log.close()
