import time
import configparser
import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from engine import trainer
from utils import *
import ast

short_name = ['01', '02', '03', '04', '05', '06', '07', '08', '09',
              '10', '11', '12', '13', '14', '15', '16', '17', '18',
              '19', '20', '21', '22', '23', '24', '25', '26', '27',
              '28', '29', '30', '31', '32']
for name in short_name:
    config_file = f'./configFiles/DEAP/s{name}.conf'
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

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log = open(args.log_file, 'w')
    log_string(log, str(args))


def concat_data(dataset_dir):
    """
    加载并将数据连接起来
    :param dataset_dir: 数据集目录
    """
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    x_combined = torch.cat([data['x_train'], data['x_val'], data['x_test']], dim=0)
    y_combined = torch.cat([data['y_train'], data['y_val'], data['y_test']], dim=0)

    return x_combined, y_combined


def main():
    # load data
    print(args.batch_size)
    X, y = concat_data(dataset_dir=args.data)
    kfold = KFold(n_splits=5, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(X)):
        X_train, X_test = X[train_ids], X[test_ids]
        y_train, y_test = y[train_ids], y[test_ids]

        # 数据标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 转换为 PyTorch 张量
        X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
        X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

        engine = trainer(args=args,
                         scaler=scaler,
                         history=args.history,
                         num_of_vertices=args.num_of_vertices,
                         in_dim=args.in_dim,
                         num_layers=args.num_layers,
                         first_layer_embedding_size=args.first_layer_embedding_size,
                         out_layer_dim=args.out_layer_dim,
                         log=log,
                         lrate=args.learning_rate,
                         device=device,
                         activation=args.activation,
                         max_grad_norm=args.max_grad_norm,
                         lr_decay=args.lr_decay,
                         temporal_emb=args.temporal_emb,
                         spatial_emb=args.spatial_emb,
                         strides=args.strides,
                         num_gcn=args.num_gcn)
        # 开始训练
        log_string(log, 'compiling model...')
        his_loss = []
        mtrain_list = []
        val_time = []
        train_time = []

        wait = 0
        val_loss_min = float('inf')
        best_model_wts = None

        for i in tqdm.tqdm(range(1, args.epochs + 1)):
            if wait >= args.patience:
                log_string(log, f'early stop at epoch: {i:04d}')
                break

            train_loss = []
            train_mae = []
            train_mape = []
            train_rmse = []
            train_acc_all = []
            train_acc_val = []
            t1 = time.time()
            for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = torch.Tensor(x).to(device)
                # [B, T, N, C]

                trainy = torch.Tensor(y[:]).to(device)
                # [B]
                loss, tmae, tmape, trmse, tacc_all, tacc_val = engine.train(trainx, trainy)
                train_loss.append(loss)
                train_mae.append(tmae)
                train_mape.append(tmape)
                train_rmse.append(trmse)
                train_acc_all.append(tacc_all)
                train_acc_val.append(tacc_val)

                if iter % args.print_every == 0:
                    logs = '\nIter: {:03d}, Train Loss: {:.4f}, MAE:{:.4f}, MAPE:{:.4f}, RMSE:{:.4f}, ' \
                           'Acc-ALL: {:.4f}, Acc-vaL: {:.4f}, lr: {}'
                    print(logs.format(iter, train_loss[-1], train_mae[-1], train_mape[-1], train_rmse[-1],
                                      train_acc_all[-1], train_acc_val[-1],
                                      engine.optimizer.param_groups[0]['lr']), flush=True)

            if args.lr_decay:
                engine.lr_scheduler.step()

            t2 = time.time()
            train_time.append(t2 - t1)

            valid_loss = []
            valid_mape = []
            valid_rmse = []
            valid_acc_all = []
            valid_acc_val = []

            s1 = time.time()
            for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                valx = torch.Tensor(x).to(device)
                # [B, T, N, C]

                valy = torch.Tensor(y[:]).to(device)
                # [B]

                vmae, vmape, vrmse, vacc_all, vacc_val = engine.evel(valx, valy)
                valid_loss.append(vmae)
                valid_mape.append(vmape)
                valid_rmse.append(vrmse)
                valid_acc_all.append(vacc_all)
                valid_acc_val.append(vacc_val)

            s2 = time.time()
            logs = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            log_string(log, logs.format(i, (s2 - s1)))

            val_time.append(s2 - s1)

            mtrain_loss = np.mean(train_loss)
            mtrain_mae = np.mean(train_mae)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)
            mtrain_acc_all = np.mean(train_acc_all)
            mtrain_acc_val = np.mean(train_acc_val)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            mvalid_acc_all = np.mean(valid_acc_all)
            mvalid_acc_val = np.mean(valid_acc_val)
            his_loss.append(mvalid_loss)
            mtrain_list.append(mtrain_loss)

            logs = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f},\n' \
                   ' Train ACC-ALL: {:.4f}, Train ACC-VAL: {:.4f},\n' \
                   ' Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f},\n' \
                   'Valid ACC-ALL: {:.4f}, Valid ACC-VAL: {:.4f}'
            log_string(log, logs.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse,
                                        mtrain_acc_all, mtrain_acc_val, mvalid_loss, mvalid_mape, mvalid_rmse,
                                        mvalid_acc_all, mvalid_acc_val))
            # os.system('nvidia-smi')

            if not os.path.exists(args.save):
                os.makedirs(args.save)

            if mvalid_loss <= val_loss_min:
                log_string(
                    log,
                    f'val loss decrease from {val_loss_min:.4f} to {mvalid_loss:.4f}, '
                    f'save model to {args.save + "exp_" + str(args.expid) + "_" + str(round(mvalid_loss, 2)) + "_best_model.pth"}'
                )
                wait = 0
                val_loss_min = mvalid_loss
                best_model_wts = engine.model.state_dict()
                torch.save(best_model_wts,
                           args.save + "exp_" + str(args.expid) + "_" + str(round(val_loss_min, 2)) + "_best_model.pth")
            else:
                wait += 1

            np.save('./history_loss' + f'_{args.expid}', his_loss)

        np.save(f'mtrain_loss_{args.expid}.npy', mtrain_list)

        log_string(log, "Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        log_string(log, "Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

        # 测试
        engine.model.load_state_dict(
            torch.load(args.save + "exp_" + str(args.expid) + "_" + str(round(val_loss_min, 2)) + "_best_model.pth"))

        outputs = []
        realy = torch.Tensor(dataloader['y_test'][:]).to(device)
        # B

        for iter, (x, y) in tqdm.tqdm(enumerate(dataloader['test_loader'].get_iterator())):
            testx = torch.Tensor(x).to(device)
            with torch.no_grad():
                preds = engine.model(testx)
                # [B]
                outputs.append(preds)

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]  # 这里是因为你在做batch的时候，可能会padding出新的sample以满足batch_size的要求

        log_string(log, "Training finished")
        log_string(log, "The valid loss on best model is " + str(round(val_loss_min, 4)))

        amae = []
        amape = []
        armse = []
        aaccu_all = []
        aaccu_val = []

        pred = yhat[:]
        real = realy[:]

        mae, mape, rmse, acc_all, acc_valence = metric(pred, real)
        logs = '最好的验证模型在测试集上Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test ACC-ALL: {:.4f}, ' \
               'Test ACC-VAL: {:.4f}'

        log_string(log, logs.format(mae, mape, rmse, acc_all, acc_valence))
        amae.append(mae)
        amape.append(mape)
        armse.append(rmse)
        aaccu_all.append(acc_all)
        aaccu_val.append(acc_valence)

        logs = '总平均测试结果, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test ACC-ALL: {:.4f}, Test ACC-VAL: {:.4f}'
        log_string(log, logs.format(np.mean(amae), np.mean(amape), np.mean(armse), np.mean(aaccu_all), np.mean(aaccu_val)))




if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()

    log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
    log.close()
