import os
import time
import argparse
from tokenize import String
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import NFMmodel
import FMmodel
import FMdata
import BPRLoss
import Procedure


def parse_args():
    #定义各项参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str,default="NFM",help="NFM or FM")
    parser.add_argument("--dataset",type=str, default="ml", help="resource of data")
    parser.add_argument("--optimizer",type=str, default="Adagrad", help="optimizer")
    parser.add_argument("--loss_type", type=str, default="bpr_loss",help="type of loss function")
    parser.add_argument("--activation_function",type=str,default="sigmoid",help="activation function")
    parser.add_argument("--lr",type=float,default=0.05,help="learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,help="the weight decay for l2 normalizaton")
    parser.add_argument("--dropout",default='[0.5, 0.2]',help="dropout rate for FM and MLP")
    parser.add_argument("--batch_size",type=int,default=128,help="batch size for training")
    parser.add_argument("--epochs",type=int,default=100,help="training epochs")
    parser.add_argument("--hidden_factor",type=int,default=64,help="predictive factors numbers in the model")
    parser.add_argument("--layers",default='[64]',help="size of layers in MLP model, '[]' is NFM-0")
    parser.add_argument("--lamda",type=float,default=0.0,help="regularizer for bilinear layers")
    parser.add_argument("--batch_norm",default=True,help="use batch_norm or not")
    parser.add_argument("--pre_train",action='store_true',default=False,help="whether use the pre-train or not")
    parser.add_argument("--out",default=True,help="save model or not")
    parser.add_argument("--gpu",type=str,default="0",help="gpu card ID")
    parser.add_argument('--verbose', type=int, default=1,help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--topks', nargs='?',default="[20]", help="@k test list")
    return parser.parse_args()

def load_data(dataset, batch_size):
    main_path = './data/{}/'.format(dataset)
    # 定义train, valid, test的路径
    train_libfm = main_path + '{}.train.libfm'.format(dataset)
    valid_libfm = main_path + '{}.validation.libfm'.format(dataset)
    test_libfm = main_path + '{}.test.libfm'.format(dataset)
    # 构造train, valid, test的数据集
    train_dataset = FMdata.FMData(train_libfm, features_map, "bpr_loss")
    valid_dataset = FMdata.FMData(valid_libfm, features_map, "bpr_loss")
    test_dataset = FMdata.FMData(test_libfm, features_map, "bpr_loss")
    # DataLoader 按照batch 打包， 然后以单线程读取数据
    train_loader = data.DataLoader(train_dataset, drop_last=True,batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = data.DataLoader(valid_dataset,batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_dataset,batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    # 定义模型的各项参数，包括轮数，批处理的数量，隐藏层的数量等
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True
   
    # 获取所有的feature, 并统计feature的数量，返回至num_features中
    features_map, num_features = FMdata.map_features(args.dataset)
    train_loader, valid_loader, test_loader= load_data(args.dataset, args.batch_size)

    
    # 构建model
    if args.model == "NFM":
        model = NFMmodel.NFM(num_features, args.hidden_factor, args.activation_function, eval(args.layers),args.batch_norm, eval(args.dropout), None)
    else:
        model = FMmodel.FM(num_features, args.hidden_factor, args.activation_function, eval(args.layers),args.batch_norm, eval(args.dropout), None)
    model.cuda()

    # optimizer，优化器
    if args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(
            model.parameters(), lr=args.lr, initial_accumulator_value=1e-8)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Momentum':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.95)

    # 损失函数 
    if args.loss_type == 'pbr_loss':
        criterion = BPRLoss(model, args.decay, args.lr, optimizer)
    elif args.loss_type == 'square_loss':
        criterion = nn.MSELoss(reduction='sum')
    else:  # log_loss
        criterion = nn.BCEWithLogitsLoss(reduction='sum')

    # 开始训练过程
    model_path = './models/'
    for epoch in range(args.epochs):
        start_time = time.time()
        if epoch % 10 == 9:
            print("[Valid]")
            Procedure.ValidTest(model, args.topks, valid_loader, args.batch_size)
        outputInfo = Procedure.BPRTrain(model, criterion, train_loader, args.batch_size)
        print("Runing Epoch {:03d} ".format(epoch) + "average_losts" + outputInfo + "costs " + time.strftime("%H: %M: %S", time.gmtime(time.time()-start_time)))

    print("[Test]")
    Procedure.ValidTest(model, args.topks, criterion, test_loader, args.batch_size)