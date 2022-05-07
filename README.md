# NFM & LightGCN 论文复现

# 1. Pretrain

## 1.1 数据来源

数据集ML-1M处理：movielen-1m， 数据来源：

[MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)

## 1.2 数据处理

- ratings.dat 数据格式：
    
    UserID::MovieID::Rating::Timestamp
    
    - UserIDs range between 1 and 6040
    - MovieIDs range between 1 and 3952
    - Ratings are made on a 5-star scale (whole-star ratings only)
    - Timestamp is represented in seconds since the epoch as returned by time(2)
    - Each user has at least 20 ratings
- 目标格式
    - 三个文件：./data/ml.test.libfm，./data/ml.validation.libfm ，./data/ml.train.libfm
    - 按照时间划分每个用户的交互，train valid test比例8:1:1
    - Label 交互过的ratings>3 为正例，反之为负例
    - 输出格式：用户 电影：评分

## 1.3 处理结果

以 ml.test.libfm 文件为例，部分输出结果如下：

![Untitled](pic/Untitled.png)

# 2. NFM & FM改写

## 2.1 载入数据

- 统计dataset中所有的feature的数量，返回至num_features中，并把feature存入到features_map中
    
    ```python
    def read_features(file, features):
    	""" Read features from the given file. """
    	i = len(features)
    	with open(file, 'r') as fd:
    		line = fd.readline()
    		while line:
    			items = line.strip().split()
    			for item in items[1:]:
    				item = item.split(':')[0]
    				if item not in features:
    					features[item] = i
    					i += 1
    			line = fd.readline()
    	return features
    
    def map_features(dataset):
    	main_path = './data/{}/'.format(dataset)
    	train_libfm = main_path + '{}.train.libfm'.format(dataset)
    	valid_libfm = main_path + '{}.validation.libfm'.format(dataset)
    	test_libfm = main_path + '{}.test.libfm'.format(dataset)
    
    	features = {}
    	features = read_features(train_libfm, features)
    	features = read_features(valid_libfm, features)
    	features = read_features(test_libfm, features)
    	print("number of features: {}".format(len(features)))
    	return features, len(features)
    ```
    
- 构造train,valid,test的数据集
    
    ```python
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
    ```
    
- 构造数据集的方式：
    
    关键代码：label : (user, features(list), ratings(list))
    
    ```python
    self.label.append((np.float32(items[0]), self.features, self.feature_values))
    ```
    

## 2.2 初始化模型

- 构建model：
    
    ```python
    # 构建model
        if args.model == "NFM":
            model = NFMmodel.NFM(num_features, args.hidden_factor, args.activation_function, eval(args.layers),args.batch_norm, eval(args.dropout), None)
        else:
            model = FMmodel.FM(num_features, args.hidden_factor, args.activation_function, eval(args.layers),args.batch_norm, eval(args.dropout), None)
        model.cuda()
    ```
    
- model 的具体结构
    - FM：线性回归模型
    - 嵌入层：embedding
    - Bi-Interaction Layer: 仅在forward中出现，原因是为一种计算方式，并非一个具体的模块
    - 多个deep layers
    - 预测层 prediction：linear
    
    附：函数对应含义
    
    - __init__: 模型初始化
    - __init_weight: 初始化weight参数
    - foward:当执行model(x)的时候，底层自动调用forward方法计算结果
- 优化器：

```python
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
```

- 损失函数：

```python
# 损失函数 
    if args.loss_type == 'pbr_loss':
        criterion = BPRLoss(model, args.decay, args.lr, optimizer)
```

PBR Loss:

```python
import torch
from torch import nn
import numpy as np

class BPRLoss:
    def __init__(self, model, decay, lr, opt):
        self.model = model
        self.weight_decay = decay
        self.lr = lr
        self.opt = opt

    def stageOne(self, users, pos, neg):
        # 计算出bpr_loss的值
        loss, reg_loss = self.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss
        # 清空上一次的梯度记录
        self.opt.zero_grad()
        #  PyTorch的反向传播(即tensor.backward())是通过autograd包来实现的，autograd包会根据tensor进行过的数学运算来自动计算其对应的梯度。
        loss.backward()
        # step()函数的作用是执行一次优化步骤，通过梯度下降法来更新参数的值。
        self.opt.step()
        return loss.cpu().item()

    def bpr_loss(self, users, pos, neg):
        # 值得注意的是，原BPR就是让正样本和负样本的得分之差尽可能达到最大
        # 这里没有负号，是尽可能的小
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
```

- 它是基于Bayesian Personalized Ranking。BPR Loss 的思想很简单，就是让正样本和负样本的得分之差尽可能达到最大，这里没有负号，是尽可能的小。
- 函数对应含义
    - __init__: 模型初始化
    - stageOne: 初始化weight参数
    - bpr_loss :当执行model(x)的时候，底层自动调用forward方法计算结果

## 2.3 train & test

### 2.3.1 train

- 使用`UniformSample_original`函数将正负样本划分出来

```python
def UniformSample_original(dataset):
    S = []
    for (user, movies, ratings) in dataset:
        positem = []
        negitem = []
        for i in range(ratings):
            if ratings[i] == 1:
                positem.append(movies[i])
            else:
                negitem.append(movies[i])
        S.append([user, positem, negitem])
    return np.array(S)
```

- 使用minibatch函数分批次处理，在bpr.stageOne中实验梯度下降，参数更新，并计算所有bpr_loss的平均值
    
    ```python
    		users = users.cuda()
        posItems = posItems()
        negItems = negItems()
        users, posItems, negItems = shuffle(users, posItems, negItems)
        aver_loss = 0
        total_batch = len(users) // batchSize + 1
        for (batch_i, (batch_users,batch_pos,batch_neg)) in enumerate(minibatch(users,posItems,negItems,batch_size=batchSize)):
            cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
            aver_loss += cri
        aver_loss = aver_loss / total_batch
        return aver_loss
    ```
    

### 2.3.2 test&valid

- 用`torch.no_grad()`表示参数不发生训练变化
- 用`minibatch`函数分批次处理，其中groundTrue表示实际为真的数据集，`getUsersRating`表示预测结果，然后使用topk获取前20个，再将多个batch的结果合并

```python
for batch_users in minibatch(users, batch_size=batchSize):
            # 获取实际为真的数据集
            groundTrue = getUserPosItems(batch_users)
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to('cuda')
            # 获取model 预测为真的数据集
            rating = model.getUsersRating(batch_users_gpu)   
            _, rating_K = torch.topk(rating, k=max_k)    
            # 将多个batch的结果合并
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
```

- 根据评分数据 和 学习出来的数据，将多个batch的recall&ndge合并在pre_results中，在`test_one_batch`函数中计算出recall&ndcg
    
    ```python
    def test_one_batch(X):  # X = zip(rating_list, groundTrue_list)
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        r = getLabel(groundTrue, sorted_items)
        recall, ndcg = [], [], []
        topks = [20]
        for k in topks:
            ret = RecallPrecision_ATk(groundTrue, r, k)
            recall.append(ret['recall'])
            ndcg.append(NDCGatK_r(groundTrue,r,k))
        return {'recall':np.array(recall), 
                'ndcg':np.array(ndcg)}
    ```
    
- 根据pre_results 中所有数据， 计算出平均recall 和 ndcg，并输出
    
    ```python
    # 对pre_results 中的所有数据，计算出平均recall 和 ndcg
            for result in pre_results:
                results['recall'] += result['recall']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            print("recall: "+ results['recall'] + "ndcg:" + results['ndcg'])
    ```
    

# 3. 资料 & 感悟

- refereces: [https://www.notion.so/751da338b1c84614b4a1e4501643a179](https://www.notion.so/751da338b1c84614b4a1e4501643a179)