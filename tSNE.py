from utils import *
import argparse
from vis_model import *
import numpy as np
import configparser
import ast
import torch
from scipy.spatial.distance import euclidean


DATASET = 's01'
dictory = f'./Topo_data/tSNE/{DATASET}/'
if not os.path.exists(dictory):
    os.makedirs(dictory)

config_file = './configFiles/DEAP/{}_test.conf'.format(DATASET)
config = configparser.ConfigParser()
config.read(config_file)

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
device = torch.device("cpu")
log = open(args.log_file, 'w')
log_string(log, str(args))


def main():
    # load data
    dataloader = load_dataset(dataset_dir=args.data,
                              normalizer=args.normalizer,
                              batch_size=args.batch_size,
                              valid_batch_size=args.batch_size,
                              test_batch_size=args.batch_size)
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

    # model.eval()
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.checkpoint).items()})
    # print(torch.load(args.checkpoint).items())
    # print(model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.checkpoint).items()}))

    log_string(log, '加载模型成功')

    x = torch.Tensor(dataloader['x_train']).to(device)
    y = torch.Tensor(dataloader['y_train']).to(device)
    numpy_y = y.numpy()
    numpy_x = x.numpy()
    print(numpy_x.shape)
    np.save(f'./Topo_data/{DATASET}-labels.npy', numpy_y)
    np.save(f'./Topo_data/{DATASET}-original.npy', numpy_x)
    # unique_rows, indices = np.unique(numpy_y, axis=0, return_index=True)
    # sample_all = np.zeros((len(x), 32, 128))
    # print(sample_all.shape, numpy_y.shape)
    # for num, label in enumerate(numpy_y):
    #     x_tsne = torch.unsqueeze(x[num, ...], dim=0)
    #     with torch.no_grad():
    #         output = model(x_tsne).numpy()
    #         output = np.squeeze(output)
    #         sample_all[num, :, :] = output
    #
    # np.save(f'./Topo_data/tSNE/{DATASET}/train_data.npy', sample_all)



if __name__ == "__main__":
    main()
    log.close()
