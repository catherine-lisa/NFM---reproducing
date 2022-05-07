import numpy as np
import torch.utils.data as data


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


class FMData(data.Dataset):
	# 构造FM数据集
	def __init__(self, file, feature_map, loss_type):
		super(FMData, self).__init__()
		self.label = []
		self.features = []
		self.feature_values = []

		with open(file, 'r') as fd:
			line = fd.readline()

			while line:
				items = line.strip().split()

				# raw为提取的数据前半部分
				raw = [item.split(':')[0] for item in items[1:]]
				# 提取电影对应编号
				self.features.append(np.array([feature_map[item] for item in raw]))
				# 冒号后的数据汇集在一起
				self.feature_values.append(np.array([item.split(':')[1] for item in items[1:]], dtype=np.float32))

				# convert labels
				if loss_type == 'bpr_loss':
					self.label.append((np.float32(items[0]), self.features, self.feature_values))
				else: # log_loss
					label = 1 if float(items[0]) > 0 else 0
					self.label.append(label)

				line = fd.readline()

	def __len__(self):
		return len(self.label)

	def __getitem__(self, idx):
		label = self.label[idx]
		features = self.features[idx]
		feature_values = self.feature_values[idx]
		return features, feature_values, label
