from statistics import mode
import torch
import BPRLoss
import numpy as np

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

def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

def minibatch(*tensors, **kwargs):

    batch_size = 128

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def BPRTrain(model, loss_class, train_loader, batchSize):
    Model = model
    Model.train()
    bpr: BPRLoss = loss_class

    S = UniformSample_original(train_loader)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

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

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

# 里面关于计算recall 和 ndcg的方式要修改
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

def getUserPosItems(users):
    user_posItems = []
    posItems = []
    for (user, movies, ratings) in users:
        for i in range(ratings):
            if ratings[i] == 1:
                posItems.append(movies[i])
        user_posItems.append((user, posItems))
    return user_posItems

def ValidTest(model, topks, dataset, batchSize):
    model = model.eval()
    results = {'precision': np.zeros(len(topks)),'recall': np.zeros(len(topks)),'ndcg': np.zeros(len(topks))}
    max_k = max(topks)
    # 确保参数不改变
    with torch.no_grad():
        users = dataset.label
        # self.label.append(zip(np.float32(items[0]), self.features, self.feature_values))
        max_k = topks
        users_list = []
        rating_list = []
        groundTrue_list = []

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

        # 根据评分数据 和 学习出来的数据，将多个batch合并在pre_results 中
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x))
        # 对pre_results 中的所有数据，计算出平均recall 和 ndcg
        for result in pre_results:
            results['recall'] += result['recall']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        print("recall: "+ results['recall'] + "ndcg:" + results['ndcg'])
