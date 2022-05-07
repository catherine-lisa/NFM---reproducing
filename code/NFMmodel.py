import torch
import torch.nn as nn
import torch.nn.functional as F


class NFM(nn.Module):
	def __init__(self, num_features, num_factors, 
		act_function, layers, batch_norm, drop_prob, pretrain_FM):
		super(NFM, self).__init__()

		# 初始化模型的指标
		self.num_features = num_features
		self.num_factors = num_factors
		self.act_function = act_function
		self.layers = layers
		self.batch_norm = batch_norm
		self.drop_prob = drop_prob
		self.pretrain_FM = pretrain_FM
		# 嵌入层 + 偏置量bias
		self.embeddings = nn.Embedding(num_features, num_factors)
		self.biases = nn.Embedding(num_features, 1)
		self.bias_ = nn.Parameter(torch.tensor([0.0]))

		# 构建FM模块
		# nn.BatchNorm1d(dim)，dim等于前一层输出的维度， BatchNorm层输出的维度也是dim。
		FM_modules = []
		if self.batch_norm:
			FM_modules.append(nn.BatchNorm1d(num_factors))		
		FM_modules.append(nn.Dropout(drop_prob[0]))
		# nn.Sequential为有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
		self.FM_layers = nn.Sequential(*FM_modules)

		# 构建deep_layer 部分
		MLP_module = []
		in_dim = num_factors
		for dim in self.layers:
			out_dim = dim
			MLP_module.append(nn.Linear(in_dim, out_dim))
			in_dim = out_dim
			# 归一化处理
			if self.batch_norm:
				MLP_module.append(nn.BatchNorm1d(out_dim))
			# 激活函数
			if self.act_function == 'relu':
				MLP_module.append(nn.ReLU())
			elif self.act_function == 'sigmoid':
				MLP_module.append(nn.Sigmoid())
			elif self.act_function == 'tanh':
				MLP_module.append(nn.Tanh())
			# dropout层，防止过拟合
			MLP_module.append(nn.Dropout(drop_prob[-1]))
		self.deep_layers = nn.Sequential(*MLP_module)

		# 预测层
		predict_size = layers[-1] if layers else num_factors
		self.prediction = nn.Linear(predict_size, 1, bias=False)
		# 初始化weight参数
		self._init_weight_()

	# 对每一层的weight做初始化处理
	def _init_weight_(self):
		""" Try to mimic the original weight initialization. """
		# 初始化嵌入层和线性回归模型的weight
		nn.init.normal_(self.embeddings.weight, std=0.01)
		nn.init.constant_(self.biases.weight, 0.0)

		# 初始化deep layer的weight 和 预测层的weight
		if len(self.layers) > 0:
			for m in self.deep_layers:
				if isinstance(m, nn.Linear):
					nn.init.xavier_normal_(m.weight)
			nn.init.xavier_normal_(self.prediction.weight)
		else:
			nn.init.constant_(self.prediction.weight, 1.0)

	def forward(self, features, feature_values):
		nonzero_embed = self.embeddings(features)
		feature_values = feature_values.unsqueeze(dim=-1)
		nonzero_embed = nonzero_embed * feature_values

		# Bi-Interaction layer
		sum_square_embed = nonzero_embed.sum(dim=1).pow(2)
		square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1)

		# FM model
		FM = 0.5 * (sum_square_embed - square_sum_embed)
		FM = self.FM_layers(FM)
		if self.layers: # have deep layers
			FM = self.deep_layers(FM)
		FM = self.prediction(FM)

		# bias addition
		feature_bias = self.biases(features)
		feature_bias = (feature_bias * feature_values).sum(dim=1)
		FM = FM + feature_bias + self.bias_
		return FM.view(-1)

	def compute(self, user):
		# self.label.append(zip(np.float32(items[0]), self.features, self.feature_values))
		features = user[:, 1]
		feature_values = user[:, 2]
		nonzero_embed = self.embeddings(features)
		feature_values = feature_values.unsqueeze(dim=-1)
		nonzero_embed = nonzero_embed * feature_values

		# Bi-Interaction layer
		sum_square_embed = nonzero_embed.sum(dim=1).pow(2)
		square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1)

		# FM model
		FM = 0.5 * (sum_square_embed - square_sum_embed)
		FM = self.FM_layers(FM)
		if self.layers: # have deep layers
			FM = self.deep_layers(FM)
		FM = self.prediction(FM)

		# bias addition
		feature_bias = self.biases(features)
		feature_bias = (feature_bias * feature_values).sum(dim=1)
		FM = FM + feature_bias + self.bias_
		return float(FM)


	def getUsersRating(self, users):
		user_rating = []
		users = users.long()
		for user in users:
			scores = self.compute(user)
			user_rating.append(user, scores)
		return user_rating

