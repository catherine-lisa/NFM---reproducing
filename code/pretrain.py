import pandas as pd
# 从文件夹中读取目标文件，并按照列划分
path = "./ml-1m/ratings.dat"
df = pd.read_table(path,sep = '::',header=None,engine='python')
df.columns = (['user_id','item_id',"rating","timestamp"])
# 按照时间戳的大小排序，打乱用户的分布，保证随机性
df = df.sort_values('timestamp', ascending=True)

# 全部将内容由df转化为list，方便后续操作
user=df['user_id'].values.tolist()
item=df['item_id'].values.tolist()
rate=df['rating'].values.tolist()

# 更新rate内容，从0-5的评分改为0,1的划分
for i in range(0, len(rate)):
    if rate[i] > 3:
        rate[i] = 1
    else:
        rate[i] = 0

# # 按照8:1:1的比例，算出每部分长度
user_len = len(user)
train_len = int(0.8*user_len)
valid_len = int(0.1*user_len)
# 用减法的方式处理，防止出现难以整除导致遗漏数据的情况
test_len = user_len - train_len - valid_len
# 按照8:1:1的比例，划分数据集
train_user, valid_user, test_user = user[:train_len], user[train_len:valid_len+train_len], user[-test_len:]
train_item, valid_item, test_item = item[:train_len], item[train_len:valid_len+train_len], item[-test_len:]
train_rate, valid_rate, test_rate = rate[:train_len], rate[train_len:valid_len+train_len], rate[-test_len:]

# train部分
# 先找到最大的用户，train_lists为存储的(item,rate)的二元元组的列表，和用户一一对应
max_user = df['user_id'].max()
train_lists = [[] for _i in range(max_user+1)]
for i in range(len(train_user)):
    train_lists[train_user[i]].append((train_item[i], train_rate[i]))

# 以每行: user movie::rate 的方式输出到文件中
f=open("./data/ml.train.libfm", "w")
for i in range(len(train_lists)):
    if len(train_lists[i]) > 0:
        f.write(str(i)+" ")
        for (item,rate) in train_lists[i]:
            f.write(str(item)+":"+str(rate)+" ") 
        f.write("\n")
f.close()
# 后面的操作方式同理

# valid 部分
max_user = df['user_id'].max()
valid_lists = [[] for _i in range(max_user+1)]
for i in range(len(valid_user)):
    valid_lists[valid_user[i]].append((valid_item[i], valid_rate[i]))

f=open("./data/ml.validation.libfm", "w")
for i in range(len(valid_lists)):
    if len(valid_lists[i]) > 0:
        f.write(str(i)+" ")
        for (item,rate) in valid_lists[i]:
            f.write(str(item)+":"+str(rate)+" ") 
        f.write("\n")
f.close()

# test 部分
max_user = df['user_id'].max()
test_lists = [[] for _i in range(max_user+1)]
for i in range(len(test_user)):
    test_lists[test_user[i]].append((test_item[i], test_rate[i]))

f=open("./data/ml.test.libfm", "w")
for i in range(len(test_lists)):
    if len(test_lists[i]) > 0:
        f.write(str(i)+" ")
        for (item,rate) in test_lists[i]:
            f.write(str(item)+":"+str(rate)+" ") 
        f.write("\n")
f.close()