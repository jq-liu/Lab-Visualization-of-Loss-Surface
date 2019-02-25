#coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import time
import datetime
import json
import sys
import sklearn.metrics
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import copy



use_cuda = torch.cuda.is_available()

def to_var(x):
	if use_cuda:
		return Variable(torch.from_numpy(x).cuda())
	else:
		return Variable(torch.from_numpy(x))

class Accuracy(object):
	def __init__(self):
		self.correct = 0
		self.total = 0
	def add(self, is_correct):
		self.total += 1
		if is_correct:
			self.correct += 1
	def get(self):
		if self.total == 0:
			return 0.0
		else:
			return float(self.correct) / self.total
	def clear(self):
		self.correct = 0
		self.total = 0 

class Config(object):
	def __init__(self):
		self.acc_NA = Accuracy()
		self.acc_not_NA = Accuracy()
		self.acc_total = Accuracy()
		self.data_path = './data'
		self.use_bag = True
		self.use_gpu = True
		self.is_training = True
		self.max_length = 120
		self.pos_num = 2 * self.max_length
		self.num_classes = 53
		self.hidden_size = 230
		self.pos_size = 5
		self.max_epoch = 15
		self.opt_method = 'SGD'
		self.learning_rate = 0.5
		self.weight_decay = 1e-5
		self.drop_prob = 0.5
		self.optimizer = None
		self.checkpoint_dir = './checkpoint'
		self.test_result_dir = './test_result'
		self.save_epoch = 1
		self.test_epoch = 1
		self.pretrain_model = None
		self.trainModel = None
		self.testModel = None
		self.pltModel = None

		# self.batch_size = 160
		self.batch_size = 500
		self.word_size = 50
		self.window_size = 3
		self.epoch_range = None
	def set_data_path(self, data_path):
		self.data_path = data_path
	def set_max_length(self, max_length):
		self.max_length = max_length
		self.pos_num = 2 * self.max_length
	def set_num_classes(self, num_classes):
		self.num_classes = num_classes
	def set_hidden_size(self, hidden_size):
		self.hidden_size = hidden_size
	def set_window_size(self, window_size):
		self.window_size = window_size
	def set_pos_size(self, pos_size):
		self.pos_size = pos_size
	def set_word_size(self, word_size):
		self.word_size = word_size
	def set_max_epoch(self, max_epoch):
		self.max_epoch = max_epoch
	def set_batch_size(self, batch_size):
		self.batch_size = batch_size
	def set_opt_method(self, opt_method):
		self.opt_method = opt_method
	def set_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate
	def set_weight_decay(self, weight_decay):
		self.weight_decay = weight_decay
	def set_drop_prob(self, drop_prob):
		self.drop_prob = drop_prob
	def set_checkpoint_dir(self, checkpoint_dir):
		self.checkpoint_dir = checkpoint_dir
	def set_test_epoch(self, test_epoch):
		self.test_epoch = test_epoch
	def set_save_epoch(self, save_epoch):
		self.save_epoch = save_epoch
	def set_pretrain_model(self, pretrain_model):
		self.pretrain_model = pretrain_model
	def set_is_training(self, is_training):
		self.is_training = is_training
	def set_use_bag(self, use_bag):
		self.use_bag = use_bag
	def set_use_gpu(self, use_gpu):
		self.use_gpu = use_gpu
	def set_epoch_range(self, epoch_range):
		self.epoch_range = epoch_range
	
	def load_train_data(self):
		print("Reading training data...")
		self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
		self.data_train_word = np.load(os.path.join(self.data_path, 'train_word.npy'))
		self.data_train_pos1 = np.load(os.path.join(self.data_path, 'train_pos1.npy'))
		self.data_train_pos2 = np.load(os.path.join(self.data_path, 'train_pos2.npy'))
		self.data_train_mask = np.load(os.path.join(self.data_path, 'train_mask.npy'))
		if self.use_bag:
			self.data_query_label = np.load(os.path.join(self.data_path, 'train_ins_label.npy'))
			self.data_train_label = np.load(os.path.join(self.data_path, 'train_bag_label.npy'))
			self.data_train_scope = np.load(os.path.join(self.data_path, 'train_bag_scope.npy'))
		else:
			self.data_train_label = np.load(os.path.join(self.data_path, 'train_ins_label.npy'))
			self.data_train_scope = np.load(os.path.join(self.data_path, 'train_ins_scope.npy'))
		print("Finish reading")
		self.train_order = list(range(len(self.data_train_label)))
		self.train_batches = int(len(self.data_train_label) / self.batch_size)
		if len(self.data_train_label) % self.batch_size != 0:
			self.train_batches += 1

	def load_test_data(self):
		print("Reading testing data...")
		self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
		self.data_test_word = np.load(os.path.join(self.data_path, 'test_word.npy'))
		self.data_test_pos1 = np.load(os.path.join(self.data_path, 'test_pos1.npy'))
		self.data_test_pos2 = np.load(os.path.join(self.data_path, 'test_pos2.npy'))
		self.data_test_mask = np.load(os.path.join(self.data_path, 'test_mask.npy'))
		if self.use_bag:
			self.data_test_label = np.load(os.path.join(self.data_path, 'test_bag_label.npy'))
			self.data_test_scope = np.load(os.path.join(self.data_path, 'test_bag_scope.npy'))
		else:
			self.data_test_label = np.load(os.path.join(self.data_path, 'test_ins_label.npy'))
			self.data_test_scope = np.load(os.path.join(self.data_path, 'test_ins_scope.npy'))
		print("Finish reading")
		self.test_batches = int(len(self.data_test_label) / self.batch_size)
		if len(self.data_test_label) % self.batch_size != 0:
			self.test_batches += 1

		self.total_recall = self.data_test_label[:, 1:].sum()

	def set_train_model(self, model):
		print("Initializing training model...")
		self.model = model
		self.trainModel = self.model(config = self)
		if self.pretrain_model != None:
			self.trainModel.load_state_dict(torch.load(self.pretrain_model))
		# self.trainModel.cuda()
		if self.optimizer != None:
			pass
		elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
			self.optimizer = optim.Adagrad(self.trainModel.parameters(), lr = self.learning_rate, lr_decay = self.lr_decay, weight_decay = self.weight_decay)
		elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
			self.optimizer = optim.Adadelta(self.trainModel.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
		elif self.opt_method == "Adam" or self.opt_method == "adam":
			self.optimizer = optim.Adam(self.trainModel.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
		else:
			self.optimizer = optim.SGD(self.trainModel.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
		print("Finish initializing")
 			 
	def set_test_model(self, model):
		print("Initializing test model...")
		self.model = model
		self.testModel = self.model(config = self)
		self.testModel.cuda()
		self.testModel.eval()
		print("Finish initializing")

	def set_plt_model(self, model):
		print("Initializing plt model...")
		self.model = model
		self.pltModel = self.model(config=self)
		self.optimizer = optim.SGD(self.pltModel.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
		if use_cuda:
			self.pltModel.cuda()
		self.pltModel.eval()
		print("Finish initializing")

	def get_train_batch(self, batch):
		input_scope = np.take(self.data_train_scope, self.train_order[batch * self.batch_size : (batch + 1) * self.batch_size], axis = 0)
		index = []
		scope = [0]
		for num in input_scope:
			index = index + list(range(num[0], num[1] + 1))
			scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
		self.batch_word = self.data_train_word[index, :]
		self.batch_pos1 = self.data_train_pos1[index, :]
		self.batch_pos2 = self.data_train_pos2[index, :]
		self.batch_mask = self.data_train_mask[index, :]	
		self.batch_label = np.take(self.data_train_label, self.train_order[batch * self.batch_size : (batch + 1) * self.batch_size], axis = 0)
		self.batch_attention_query = self.data_query_label[index]
		self.batch_scope = scope
		return len(input_scope)

	
	def get_test_batch(self, batch):
		input_scope = self.data_test_scope[batch * self.batch_size : (batch + 1) * self.batch_size]
		index = []
		scope = [0]
		for num in input_scope:
			index = index + list(range(num[0], num[1] + 1))
			scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
		self.batch_word = self.data_test_word[index, :]
		self.batch_pos1 = self.data_test_pos1[index, :]
		self.batch_pos2 = self.data_test_pos2[index, :]
		self.batch_mask = self.data_test_mask[index, :]
		self.batch_scope = scope
		return len(input_scope)

	def train_one_step(self):
		self.trainModel.embedding.word = to_var(self.batch_word)
		self.trainModel.embedding.pos1 = to_var(self.batch_pos1)
		self.trainModel.embedding.pos2 = to_var(self.batch_pos2)
		self.trainModel.encoder.mask = to_var(self.batch_mask)
		self.trainModel.selector.scope = self.batch_scope
		self.trainModel.selector.attention_query = to_var(self.batch_attention_query)
		self.trainModel.selector.label = to_var(self.batch_label)
		self.trainModel.classifier.label = to_var(self.batch_label)
		self.optimizer.zero_grad()
		loss, _output = self.trainModel()
		loss.backward()
		self.optimizer.step()
		for i, prediction in enumerate(_output):
			if self.batch_label[i] == 0:
				self.acc_NA.add(int(prediction) == self.batch_label[i])
			else:
				self.acc_not_NA.add(int(prediction) == self.batch_label[i])
			self.acc_total.add(int(prediction) == self.batch_label[i])
		return loss.item()

	def test_one_step(self):
		self.testModel.embedding.word = to_var(self.batch_word)
		self.testModel.embedding.pos1 = to_var(self.batch_pos1)
		self.testModel.embedding.pos2 = to_var(self.batch_pos2)
		self.testModel.encoder.mask = to_var(self.batch_mask)
		self.testModel.selector.scope = self.batch_scope
		return self.testModel.test()

	def train(self):
		# self.trainModel.load_state_dict(torch.load('./checkpoint/CNN_AVE-19'))
		if not os.path.exists(self.checkpoint_dir):
			os.mkdir(self.checkpoint_dir)
		best_auc = 0.0
		best_p = None
		best_r = None
		best_epoch = 0
		for epoch in range(self.max_epoch):
			print('Epoch ' + str(epoch) + ' starts...')
			self.acc_NA.clear()
			self.acc_not_NA.clear()
			self.acc_total.clear()
			np.random.shuffle(self.train_order)
			print('batches = %d' % self.train_batches)
			for batch in range(self.train_batches):
				self.get_train_batch(batch)
				loss = self.train_one_step()
				# if batch % 200 ==0:
				# 	# self.plt_in_epoch(epoch, batch)
				# 	path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch) + '-BS=' + str(batch))
				# 	torch.save(self.trainModel.state_dict(), path)
				time_str = datetime.datetime.now().isoformat()
				sys.stdout.write("epoch %d step %d time %s | loss: %f, NA accuracy: %f, not NA accuracy: %f, total accuracy: %f\r" % (epoch, batch, time_str, loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()))	
				sys.stdout.flush()
			if (epoch + 1) % self.save_epoch == 0:
				print('Epoch ' + str(epoch) + ' has finished')
				print('Saving model...')
				path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
				torch.save(self.trainModel.state_dict(), path)
				print('Have saved model to ' + path)
			if (epoch + 1) % self.test_epoch == 0:
				self.testModel = self.trainModel
				auc, pr_x, pr_y = self.test_one_epoch()
				if auc > best_auc:
					best_auc = auc
					best_p = pr_x
					best_r = pr_y
					best_epoch = epoch
		print("Finish training")
		print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
		print("Storing best result...")
		if not os.path.isdir(self.test_result_dir):
			os.mkdir(self.test_result_dir)
		np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_x.npy'), best_p)
		np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_y.npy'), best_r)
		print("Finish storing")
	def test_one_epoch(self):
		test_score = []
		for batch in tqdm(range(self.test_batches)):
			self.get_test_batch(batch)
			batch_score = self.test_one_step()
			test_score = test_score + batch_score
		test_result = []
		for i in range(len(test_score)):
			for j in range(1, len(test_score[i])):
				test_result.append([self.data_test_label[i][j], test_score[i][j]])
		test_result = sorted(test_result, key = lambda x: x[1])
		test_result = test_result[::-1]
		pr_x = []
		pr_y = []
		correct = 0
		for i, item in enumerate(test_result):
			correct += item[0]
			pr_y.append(float(correct) / (i + 1))
			pr_x.append(float(correct) / self.total_recall)
		auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
		print("auc: ", auc)
		return auc, pr_x, pr_y
	def test(self):
		best_epoch = None
		best_auc = 0.0
		best_p = None
		best_r = None
		for epoch in self.epoch_range:
			path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
			if not os.path.exists(path):
				continue
			print("Start testing epoch %d" % (epoch))
			self.testModel.load_state_dict(torch.load(path))
			auc, p, r = self.test_one_epoch()
			if auc > best_auc:
				best_auc = auc
				best_epoch = epoch
				best_p = p
				best_r = r
			print("Finish testing epoch %d" % (epoch))
		print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
		print("Storing best result...")
		if not os.path.isdir(self.test_result_dir):
			os.mkdir(self.test_result_dir)
		np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_x.npy'), best_p)
		np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_y.npy'), best_r)
		print("Finish storing")


	def crunch(self, coordinates, model_path, w, d):
		# self.pltModel.load_state_dict(torch.load(model_path))
		# w = copy.deepcopy(self.get_weights())
		# d = copy.deepcopy(self.get_direction())
		print('Start computing...')
		xcoordinates = coordinates[0]
		ycoordinates = coordinates[1]
		shape = (len(xcoordinates), len(ycoordinates))
		losses = -np.ones(shape=shape)
		inds, coords = self.get_job_indices(losses, xcoordinates, ycoordinates)
		criterion = nn.CrossEntropyLoss()
		np.random.shuffle(self.train_order)
		self.get_train_batch(0)
		for count, ind in enumerate(inds):
			coord = coords[count]
			self.set_weights(w, d, coord)
			loss = self.eval_loss()
			print('count = %d, ind = %d, loss = %f' % (count, ind, loss))
			losses.ravel()[ind] = loss
		return losses

	def eval_loss(self):
		with torch.no_grad():
			self.pltModel.embedding.word = to_var(self.batch_word)
			self.pltModel.embedding.pos1 = to_var(self.batch_pos1)
			self.pltModel.embedding.pos2 = to_var(self.batch_pos2)
			self.pltModel.encoder.mask = to_var(self.batch_mask)
			self.pltModel.selector.scope = self.batch_scope
			self.pltModel.selector.attention_query = to_var(self.batch_attention_query)
			self.pltModel.selector.label = to_var(self.batch_label)
			self.pltModel.classifier.label = to_var(self.batch_label)
			self.optimizer.zero_grad()
			loss, _output = self.pltModel()
		return loss.item()

	def get_weights(self):
		return [self.pltModel.embedding.word_embedding.weight.data, self.pltModel.embedding.pos1_embedding.weight.data, self.pltModel.embedding.pos2_embedding.weight.data, self.pltModel.encoder.cnn.cnn.weight.data]

	def get_diff_weights(self, weights, weights2):
		""" Produce a direction from 'weights' to 'weights2'."""
		return [w2 - w for (w, w2) in zip(weights, weights2)]

	def set_weights(self, weights, directions=None, step=None):
		dx = directions[0]
		dy = directions[1]
		changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]
		self.pltModel.embedding.word_embedding.weight.data = weights[0] + changes[0]
		self.pltModel.embedding.pos1_embedding.weight.data = weights[1] + changes[1]
		self.pltModel.embedding.pos2_embedding.weight.data = weights[2] + changes[2]
		self.pltModel.encoder.cnn.cnn.weight.data = weights[3] + changes[3]

	def get_direction(self):
		weights = self.get_weights()
		xdirection = [torch.randn(w.size()) for w in weights]
		self.normalize_directions(xdirection, weights)
		ydirection = [torch.randn(w.size()) for w in weights]
		self.normalize_directions(ydirection, weights)
		return [xdirection, ydirection]

	def normalize_directions(self, direction, weights):
		for d, w in zip(direction, weights):
			if d.dim() <= 1:
				d.fill_(0)
			else:
				for di, we in zip(d, w):
					di.mul_(we.norm()/(di.norm() + 1e-10))

	def get_job_indices(self, losses, xcoordinates, ycoordinates):
		inds = np.array(range(losses.size))
		inds = inds[losses.ravel() <= 0]
		if ycoordinates is not None:
			xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
			s1 = xcoord_mesh.ravel()[inds]
			s2 = ycoord_mesh.ravel()[inds]
			return inds, np.c_[s1,s2]
		else:
			return inds, xcoordinates.ravel()[inds]

	def plot_2d_contour(self, op, coordinates, looses, vmin=0.1, vmax=10, vlevel=0.5):
		x = coordinates[0]
		y = coordinates[1]
		X, Y = np.meshgrid(x, y)
		Z = looses

		fig = plt.figure()
		CS = plt.contour(X, Y, Z, cmpa='summer', levels=np.arange(vmin, vmax, vlevel))
		plt.clabel(CS, inline=1, fontsize=8)
		fig.savefig(op + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
		# plt.show()
		print("PLT END!")

	#########################
	# plot trajectory
	#########################
	def project_trajectory(self, dir_file, w, model_files):

		# read directions and convert them to vectors
		xdirection = dir_file[0]
		ydirection = dir_file[1]
		directions = [xdirection, ydirection]
		dx = self.nplist_to_tensor(directions[0])
		dy = self.nplist_to_tensor(directions[1])

		xcoord, ycoord = [], []
		for model_file in model_files:
			self.pltModel.load_state_dict(torch.load(model_file))
			self.pltModel.eval()
			w2 = self.get_weights()
			d = self.get_diff_weights(w, w2)
			d = self.tensorlist_to_tensor(d)

			x, y = self.project_2D(d, dx, dy)
			print("%s  (%.4f, %.4f)" % (model_file, x, y))

			xcoord.append(x)
			ycoord.append(y)
		proj_file = [np.array(xcoord), np.array(ycoord)]

		return proj_file


	def project_2D(self, d, dx, dy):
		""" Project vector d to the plane spanned by dx and dy.

			Args:
				d: vectorized weights
				dx: vectorized direction
				dy: vectorized direction
			Returns:
				x, y: the projection coordinates
		"""
		x = self.project_1D(d, dx)
		y = self.project_1D(d, dy)
		return x, y

	def project_1D(self, w, d):
		""" Project vector w to vector d and get the length of the projection.

			Args:
				w: vectorized weights
				d: vectorized direction

			Returns:
				the projection scalar
		"""
		assert len(w) == len(d), 'dimension does not match for w and '
		scale = torch.dot(w, d) / d.norm()
		return scale.item()

	def nplist_to_tensor(self, nplist):
		v = []
		for d in nplist:
			w = torch.tensor(d * np.float64(1.0))
			# Ignoreing the scalar values (w.dim() = 0).
			if w.dim() > 1:
				v.append(w.view(w.numel()))
			elif w.dim() == 1:
				v.append(w)
		return torch.cat(v)

	def setup_PCA_directions(self, model_files, w):
		"""
			Find PCA directions for the optimization path from the initial model
			to the final trained model.

			Returns:
				dir_name: the h5 file that stores the directions.
		"""

		# load models and prepare the optimization path matrix
		matrix = []

		for model_file in model_files:
			print(model_file)
			self.pltModel.load_state_dict(torch.load(model_file))
			self.pltModel.eval()
			w2 = self.get_weights()
			d = self.get_diff_weights(w, w2)
			d = self.tensorlist_to_tensor(d)
			matrix.append(d.numpy())

		# Perform PCA on the optimization path matrix
		print("Perform PCA on the models")
		pca = PCA(n_components=2)
		pca.fit(np.array(matrix))
		pc1 = np.array(pca.components_[0])
		pc2 = np.array(pca.components_[1])
		print("angle between pc1 and pc2: %f" % self.cal_angle(pc1, pc2))
		print("pca.explained_variance_ratio_: %s" % str(pca.explained_variance_ratio_))

		# convert vectorized directions to the same shape as models to save in h5 file.
		xdirection = self.npvec_to_tensorlist(pc1, w)
		ydirection = self.npvec_to_tensorlist(pc2, w)
		dir_file = [xdirection, ydirection, pca.explained_variance_ratio_, pca.singular_values_, pca.explained_variance_]
		return dir_file

	def tensorlist_to_tensor(self, weights):
		""" Concatnate a list of tensors into one tensor.

			Args:
				weights: a list of parameter tensors, e.g. net_plotter.get_weights(net).

			Returns:
				concatnated 1D tensor
		"""
		return torch.cat([w.view(w.numel()) if w.dim() > 1 else torch.FloatTensor(w) for w in weights])

	def cal_angle(self, vec1, vec2):
		""" Calculate cosine similarities between two torch tensors or two ndarraies
			Args:
				vec1, vec2: two tensors or numpy ndarraies
		"""
		if isinstance(vec1, torch.Tensor) and isinstance(vec1, torch.Tensor):
			return torch.dot(vec1, vec2) / (vec1.norm() * vec2.norm()).item()
		elif isinstance(vec1, np.ndarray) and isinstance(vec2, np.ndarray):
			return np.ndarray.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

	def npvec_to_tensorlist(self, direction, params):
		""" Convert a numpy vector to a list of tensors with the same shape as "params".

			Args:
				direction: a list of numpy vectors, e.g., a direction loaded from h5 file.
				base: a list of parameter tensors from net

			Returns:
				a list of tensors with the same shape as base
		"""
		if isinstance(params, list):
			w2 = copy.deepcopy(params)
			idx = 0
			for w in w2:
				w.copy_(torch.tensor(direction[idx:idx + w.numel()]).view(w.size()))
				idx += w.numel()
			assert (idx == len(direction))
			return w2
		else:
			s2 = []
			idx = 0
			for (k, w) in params.items():
				s2.append(torch.Tensor(direction[idx:idx + w.numel()]).view(w.size()))
				idx += w.numel()
			assert (idx == len(direction))
			return s2

	def plot_trajectory(self, dir_file, proj_file):
		""" Plot optimization trajectory on the plane spanned by given directions."""

		fig = plt.figure()
		plt.plot(proj_file[0], proj_file[1], marker='.')
		plt.tick_params('y', labelsize='x-large')
		plt.tick_params('x', labelsize='x-large')

		ratio_x = dir_file[2][0]
		ratio_y = dir_file[2][1]
		plt.xlabel('1st PC: %.2f %%' % (ratio_x * 100), fontsize='xx-large')
		plt.ylabel('2nd PC: %.2f %%' % (ratio_y * 100), fontsize='xx-large')
		fig.savefig('proj_file' + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
		# plt.show()

	def plot_contour_trajectory(self, surf_file, dir_file, proj_file, coordinates, surf_path='./loss_surface',
								vmin=0.1, vmax=10, vlevel=0.5):
		"""2D contour + trajectory"""

		x = coordinates[0]
		y = coordinates[1]
		X, Y = np.meshgrid(x, y)
		Z = surf_file

		fig = plt.figure()
		CS1 = plt.contour(X, Y, Z, levels=np.arange(vmin, vmax, vlevel))
		CS2 = plt.contour(X, Y, Z, levels=np.logspace(0.01, 0.08, num=8))

		# plot trajectories
		plt.plot(proj_file[0], proj_file[1], marker='.')


		# add PCA notes
		plt.clabel(CS1, inline=1, fontsize=6)
		plt.clabel(CS2, inline=1, fontsize=6)
		fig.savefig(surf_path + '2D-contour_proj.pdf', dpi=300, bbox_inches='tight', format='pdf')
		plt.show()

