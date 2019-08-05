"""
Author: Xiaocong Du
Description: 
Title: Single-Net Continual Learning with Progressive Segmented Training (PST)
"""

import torch
import os
import random
import re
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import pickle
import torch.nn as nn
import numpy as np
from args import parser
from scipy.spatial.distance import cdist
from resnet import *
from utils_tool import progress_bar
args = parser.parse_args()


class ContinualNN(object):
	def __init__(self):
		self.batch_size = args.batch_size
		self.model_path = '../model_library/'
		
		
		if not os.path.exists('../results/'):
			os.mkdir('../results')
		if not os.path.exists('../mask_library/'):
			os.mkdir('../mask_library')

	def initial_single_network(self, init_weights = True):
		if args.dataset == 'cifar100':
			self.net = resnet32()
			print('Network: ResNet32, init_weights=True')
		return self.net


	def initialization(self, lr, lr_step_size, weight_decay):
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.net = self.net.to(self.device)
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.net.parameters(),  lr = lr,  momentum = 0.9, weight_decay = weight_decay)
		self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = lr_step_size, gamma= args.lr_gamma)


	def create_task(self):
		if args.dataset == 'cifar10':
			total_classes = 10
		elif args.dataset == 'cifar100':
			total_classes = 100
		# random select label
		a = list(range(0, total_classes))
		if args.shuffle:
			random.seed(args.seed)
			random.shuffle(a)
		else:
			a = a
		task_list = []
		for i in range(0, len(a), args.classes_per_task):
			task_list.append(a[i:i + args.classes_per_task])
		self.task_list = task_list
		total_num_task = int(total_classes / args.classes_per_task)
		return self.task_list, total_num_task


	def train(self, epoch, trainloader):
		if args.resume:
			# Load checkpoint.
			print('==> Resuming from checkpoint..')
			assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
			checkpoint = torch.load('./checkpoint/ckpt.t7')
			self.net.load_state_dict(checkpoint['net'])
			best_acc = checkpoint['acc']
			start_epoch = checkpoint['epoch']

		print('\nEpoch: %d lr: %s' % (epoch, self.scheduler.get_lr()))
		self.scheduler.step()
		self.net.train()
		train_loss = 0.0
		correct = 0
		total = 0

		for batch_idx, (inputs, targets) in enumerate(trainloader):
			inputs, targets = inputs.to(self.device), targets.to(self.device)
			inputs_var = Variable(inputs)
			targets_var = Variable(targets)

			self.optimizer.zero_grad()

			outputs,_ = self.net(inputs_var)
			loss = self.criterion(outputs, targets_var)

			loss.backward()
			self.optimizer.step()

			train_loss += loss.item()
			_, predicted = outputs.max(1) # outputs.shape: (batch, classes)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			self.loss = train_loss
			progress_bar(batch_idx, len(trainloader), 'Loss:%.3f|Acc:%.3f%% (%d/%d)--Train' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
		return correct/total


	def train_fc(self, epoch, trainloader):
		for name, param in self.net.named_parameters():
			if re.search('conv', name) or re.search('bn', name):
				param.requires_grad = False
			elif re.search('linear', name):
				param.requires_grad = True

		self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

		print('\nEpoch: %d lr: %s' % (epoch, self.scheduler.get_lr()))
		self.scheduler.step()
		self.net.train()
		train_loss = 0.0
		correct = 0
		total = 0

		for batch_idx, (inputs, targets) in enumerate(trainloader):
			inputs, targets = inputs.to(self.device), targets.to(self.device)
			inputs_var = Variable(inputs)
			targets_var = Variable(targets)
			self.optimizer.zero_grad()
			outputs, _ = self.net(inputs_var)
			loss = self.criterion(outputs, targets_var)
			loss.backward()
			self.optimizer.step()
			train_loss += loss.item()
			_, predicted = outputs.max(1)  # outputs.shape: (batch, classes)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			self.loss = train_loss
			progress_bar(batch_idx, len(trainloader), 'Loss:%.3f|Acc:%.3f%% (%d/%d)--Train' % (
			train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
		return correct / total


	def initialize_fc(self):
		param_old = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
		for layer_name, param in self.net.state_dict().items():
			if re.search('linear', layer_name):
				param_old[layer_name] = nn.init.normal_(param.clone(), 0, 0.01)
			else:
				param_old[layer_name] = param.clone()
		self.net.load_state_dict(param_old)


	def test(self, testloader):
		self.net.eval()
		test_loss = 0.0
		correct = 0
		total = 0
		with torch.no_grad():
			for batch_idx, (inputs, targets) in enumerate(testloader):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				inputs = Variable(inputs)
				targets = Variable(targets)

				outputs, _ = self.net(inputs)
				loss = self.criterion(outputs, targets)

				test_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()

				progress_bar(batch_idx, len(testloader), 'Loss:%.3f|Acc:%.3f%% (%d/%d)--Test'
							 % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

		return correct/total


	def save_checkpoint(self, epoch, acc):
		print('Saving checkpoint..')
		if not os.path.isdir('../checkpoint'):
			os.mkdir('../checkpoint')
		PATH = '../checkpoint/ckpt_epoch{}_accu{}.pt'.format(epoch, acc)
		torch.save({
			'epoch': epoch,
			'model_state_dict': self.net.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'loss': self.loss,
		}, PATH)


	def save_model(self, model_id, task_id):
		if not os.path.isdir(self.model_path):
		   os.mkdir(self.model_path)
		file_name = "Task{}_model{}_classes{}.pickle".format(task_id, model_id, args.classes_per_task)
		path = os.path.join(self.model_path, file_name)
		pickle.dump(self.net.state_dict(), open(path, 'wb'))

		print('save_model: model {} of Task {} is saved in [{}]\n'.format(model_id, task_id, path) )


	def load_model(self, model_id, task_id, to_net):
		file_name = "Task{}_model{}_classes{}.pickle".format(task_id, model_id, args.classes_per_task)
		path = os.path.join(self.model_path, file_name)
		param_model_dict = pickle.load(open(path, 'rb'))
		to_net.load_state_dict(param_model_dict)
		print('load_model: Loading {}....'.format(path) )



	def load_model_random_initial(self, save_mask_file, save_mask_fileR, model_id, task_id):
		try:
			mask_dict = pickle.load(open(save_mask_file, "rb"))
			mask_reverse_dict = pickle.load(open(save_mask_fileR, "rb"))
		except TypeError:
			mask_dict = save_mask_file
			mask_reverse_dict = save_mask_fileR

		param_random = OrderedDict([(k, None) for k in self.net.state_dict().keys()])

		for layer_name, param in self.net.state_dict().items():
			param_random[layer_name] = Variable(param.type(torch.cuda.FloatTensor).clone(), requires_grad = True)

		param_processed = OrderedDict([(k,None) for k in self.net.state_dict().keys()])

		self.load_model(model_id, task_id, self.net)

		for layer_name, param_model in self.net.state_dict().items():
			param_model = Variable(param_model.type(torch.cuda.FloatTensor), requires_grad = True)
			if layer_name in mask_dict.keys(): # if layer_name from networkA, load model with mask, randomly initialize the rest
				param_processed[layer_name] = Variable(torch.mul(param_model, mask_dict[layer_name]) + torch.mul(param_random[layer_name], mask_reverse_dict[layer_name]), requires_grad = True)
			else: # if new network, randomly initialize
				param_processed[layer_name] = Variable(param_random[layer_name], requires_grad = True)
		assert param_processed[layer_name].get_device() == self.net.state_dict()[layer_name].get_device(), "parameter and net are not in same device"

		self.net.load_state_dict(param_processed)
		print('load_model_random_initial: Random initialize masked weights.\n')
		return self.net

	def convert_list_to_dict(self, gradient_list, threshold_list, mask_file, taylor_list): # test drift range of the rest parameters
		threshold_dict = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
		gradient_dict = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
		mask_dict = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
		mask_R_dict = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
		taylor_dict = OrderedDict([(k, None) for k in self.net.state_dict().keys()])

		idx = 0

		mask_list = []
		mask_list_R = []
		for i in range(len(mask_file[0])):
			mask_list.append(torch.from_numpy(mask_file[0][i]).type(torch.cuda.FloatTensor))
			mask_list_R.append(torch.from_numpy(mask_file[1][i]).type(torch.cuda.FloatTensor))

		for layer_name, param in self.net.state_dict().items():
			# print(layer_name, param.shape)
			threshold_dict[layer_name] = threshold_list[idx]
			gradient_dict[layer_name] = gradient_list[idx]
			mask_dict[layer_name] = mask_list[idx]
			mask_R_dict[layer_name] = mask_list_R[idx]
			taylor_dict[layer_name] = taylor_list[idx]
			idx += 1
		# for i, key in enumerate(mask_dict): # check if threshold loading into correct dictionary
		#     assert threshold_list[i] == threshold_dict[key], 'Threshold loading incorrect'
		#     assert taylor_list[i].all() == taylor_dict[key].all(), 'Taylor loading incorrect'
		#     assert gradient_list[i].all() == gradient_dict[key].all(), 'Gradient loading incorrect'
		print('Several lists are converted into dictionaries (in torch.cuda)\n\n')
		return  gradient_dict, threshold_dict, mask_dict, mask_R_dict, taylor_dict



	def train_with_frozen_filter(self, epoch, trainloader, mask_dict, mask_dict_R):
		param_old_dict = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
		for layer_name, param in self.net.state_dict().items():
			param_old_dict[layer_name] = param.clone()

		print('\nEpoch: %d lr: %s' % (epoch, self.scheduler.get_lr()))
		self.scheduler.step()
		self.net.train()
		train_loss = 0.0
		correct = 0
		total = 0

		for batch_idx, (inputs, targets) in enumerate(trainloader):

			inputs, targets = inputs.to(self.device), targets.to(self.device)

			inputs_var = Variable(inputs)
			targets_var = Variable(targets)
			self.optimizer.zero_grad()
			outputs,_ = self.net(inputs_var)
			loss = self.criterion(outputs, targets)
			loss.backward()
			self.optimizer.step()

			train_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

			# apply mask
			param_processed = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
			for layer_name, param_new in self.net.state_dict().items():
				param_new = param_new.type(torch.cuda.FloatTensor)
				param_old_dict[layer_name] = param_old_dict[layer_name].type(torch.cuda.FloatTensor)

				if re.search('conv', layer_name):
					param_processed[layer_name] = Variable(torch.mul(param_old_dict[layer_name], mask_dict[layer_name]) +
														   torch.mul(param_new, mask_dict_R[layer_name]), requires_grad=True)

				elif re.search('shortcut', layer_name):
					if len(param_new.shape) == 4:  # conv in shortcut
						param_processed[layer_name] = Variable(torch.mul(param_old_dict[layer_name], mask_dict[layer_name]) +
														   torch.mul(param_new, mask_dict_R[layer_name]), requires_grad=True)
					else:
						param_processed[layer_name] = Variable(param_new, requires_grad=True)
				elif re.search('linear', layer_name):
					param_processed[layer_name] = Variable(torch.mul(param_old_dict[layer_name], mask_dict[layer_name]) +
						torch.mul(param_new, mask_dict_R[layer_name]), requires_grad=True)

				else:
					param_processed[layer_name] = Variable(param_new, requires_grad=True)  # num_batches_tracked
					# raise ValueError('some parameters are skipped, plz check {}'.format(layer_name))  # num_batches_tracked
			self.net.load_state_dict(param_processed)
			progress_bar(batch_idx, len(trainloader), 'Loss:%.3f|Acc:%.3f%% (%d/%d)--Train' % (
			train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
		return correct / total


	def test_multihead(self, task_id, testloader):
		self.net.eval()
		test_loss = 0.0
		correct = 0
		total = 0
		with torch.no_grad():
			for batch_idx, (inputs, targets) in enumerate(testloader):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				inputs = Variable(inputs)
				targets = Variable(targets)

				outputs, _ = self.net(inputs)
				loss = self.criterion(outputs, targets)

				test_loss += loss.item()
				_, predicted = outputs[:, args.classes_per_task*task_id:args.classes_per_task*(task_id+1)].max(1)
				total += targets.size(0)
				correct += (predicted+args.classes_per_task*task_id).eq(targets).sum().item()

				progress_bar(batch_idx, len(testloader), 'Loss:%.3f|Acc:%.3f%% (%d/%d)--Test'
							 % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

		return correct/total


	def sensitivity_rank_taylor_filter(self, threshold):
		mask_list_4d = []
		mask_list_R_4d = []
		threshold_list = []
		gradient_list = []
		weight_list = []
		taylor_list = []
		i = 0
		print("Obtain top {} position according to {} ........".format(threshold, args.score))

		for m in self.net.modules():
			# print(m, m.weight.data.shape)
			if type(m) != nn.Sequential and i != 0:
				if isinstance(m, nn.Conv2d):
					total_param = m.weight.data.shape[0]
					weight_copy = m.weight.data.abs().clone().cpu().numpy()
					grad_copy = m.weight.grad.data.abs().clone().cpu().numpy()
					
					taylor = np.sum(weight_copy*grad_copy, axis=(1, 2, 3))

					num_keep = int(total_param * threshold)
					arg_max = np.argsort(taylor) # Returns the indices sort an array. small->big
					arg_max_rev = arg_max[::-1][:num_keep]
					thre = taylor[arg_max_rev[-1]]
					mask = np.zeros(weight_copy.shape)
					mask_R = np.ones(weight_copy.shape)
					mask[arg_max_rev.tolist(), :, :, :] = 1.0
					mask_R[arg_max_rev.tolist(), :, :, :] = 0.0

					mask_list_4d.append(mask)  # 0 is more
					mask_list_R_4d.append(mask_R)  # 1 is more
					threshold_list.append(thre)
					gradient_list.append(m.weight.grad.data.clone().cpu().numpy())
					weight_list.append(m.weight.data.clone().cpu().numpy())
					taylor_list.append(taylor)

				elif isinstance(m, nn.BatchNorm2d):
					# bn weight
					total_param = m.weight.data.shape[0]
					weight_copy = m.weight.data.abs().clone().cpu().numpy()
					grad_copy = m.weight.grad.data.abs().clone().cpu().numpy()
					
					taylor = weight_copy*grad_copy  #

					num_keep = int(total_param * threshold)
					arg_max = np.argsort(taylor)  # Returns the indices sort an array. small->big
					arg_max_rev = arg_max[::-1][:num_keep]
					thre = taylor[arg_max_rev[-1]]
					mask = np.zeros(weight_copy.shape)
					mask_R = np.ones(weight_copy.shape)
					mask[arg_max_rev.tolist()] = 1.0
					mask_R[arg_max_rev.tolist()] = 0.0
					mask_list_4d.append(mask)  # 0 is more
					mask_list_R_4d.append(mask_R)
					threshold_list.append(thre)
					gradient_list.append(m.weight.grad.data.clone().cpu().numpy())
					weight_list.append(m.weight.data.clone().cpu().numpy())
					taylor_list.append(taylor)

					##bn bias
					total_param = m.bias.data.shape[0]
					weight_copy = m.bias.data.abs().clone().cpu().numpy()
					grad_copy = m.bias.grad.data.abs().clone().cpu().numpy()
					
					taylor = weight_copy*grad_copy  #

					num_keep = int(total_param * threshold)
					arg_max = np.argsort(taylor)  # Returns the indices sort an array. small->big
					arg_max_rev = arg_max[::-1][:num_keep]
					thre = taylor[arg_max_rev[-1]]
					mask = np.zeros(weight_copy.shape)
					mask_R = np.ones(weight_copy.shape)
					mask[arg_max_rev.tolist()] = 1.0
					mask_R[arg_max_rev.tolist()] = 0.0
					mask_list_4d.append(mask)
					mask_list_R_4d.append(mask_R)
					threshold_list.append(thre)
					gradient_list.append(m.bias.grad.data.clone().cpu().numpy())
					weight_list.append(m.bias.data.clone().cpu().numpy())
					taylor_list.append(taylor)

					# # running_mean
					total_param = m.running_mean.data.shape[0]
					weight_copy = m.running_mean.data.abs().clone().cpu().numpy()
					num_keep = int(total_param * threshold)
					arg_max = np.argsort(taylor)  # Returns the indices sort an array. small->big
					arg_max_rev = arg_max[::-1][:num_keep]
					thre = taylor[arg_max_rev[-1]]
					mask = np.zeros(weight_copy.shape)
					mask_R = np.ones(weight_copy.shape)
					mask[arg_max_rev.tolist()] = 1.0
					mask_R[arg_max_rev.tolist()] = 0.0
					mask_list_4d.append(mask)
					mask_list_R_4d.append(mask_R)
					threshold_list.append(thre)
					gradient_list.append(m.bias.grad.data.clone().cpu().numpy())
					weight_list.append(m.bias.data.clone().cpu().numpy())
					taylor_list.append(taylor)

					total_param = m.running_var.data.shape[0]
					weight_copy = m.running_var.data.abs().clone().cpu().numpy()
					taylor = weight_copy  # * weight_copy
					num_keep = int(total_param * threshold)
					arg_max = np.argsort(taylor)  # Returns the indices sort an array. small->big
					arg_max_rev = arg_max[::-1][:num_keep]
					thre = taylor[arg_max_rev[-1]]
					mask = np.zeros(weight_copy.shape)
					mask_R = np.ones(weight_copy.shape)
					mask[arg_max_rev.tolist()] = 1.0
					mask_R[arg_max_rev.tolist()] = 0.0
					mask_list_4d.append(mask)
					mask_list_R_4d.append(mask_R)
					threshold_list.append(thre)
					gradient_list.append(m.bias.grad.data.clone().cpu().numpy())
					weight_list.append(m.bias.data.clone().cpu().numpy())
					taylor_list.append(taylor)

					if torch.__version__ == '1.0.1.post2': # torch 1.0 bn.num_tracked
						mask_list_4d.append(np.zeros(1))
						mask_list_R_4d.append(np.zeros(1))
						threshold_list.append(np.zeros(1))
						gradient_list.append(np.zeros(1))
						weight_list.append(np.zeros(1))
						taylor_list.append(taylor)

				elif isinstance(m, nn.Linear): # neuron-wise
					#linear weight
					weight_copy = m.weight.data.abs().clone().cpu().numpy()
					grad_copy = m.weight.grad.data.abs().clone().cpu().numpy()

					taylor = np.sum(weight_copy*grad_copy, axis = 1)

					num_keep = int(m.weight.data.shape[0] * threshold)
					arg_max = np.argsort(taylor)  # Returns the indices that would sort an array. small->big
					arg_max_rev = arg_max[::-1][:num_keep]
					thre = taylor[arg_max_rev[-1]]
					mask = np.zeros(weight_copy.shape)
					mask_R = np.ones(weight_copy.shape)
					mask[arg_max_rev.tolist(), :] = 1.0
					mask_R[arg_max_rev.tolist(), :] = 0.0
					mask_list_4d.append(mask)  # 0 is more
					mask_list_R_4d.append(mask_R)  # 1 is more
					threshold_list.append(thre)
					gradient_list.append(m.weight.grad.data.clone())
					weight_list.append(m.weight.data.clone())
					taylor_list.append(taylor)

					# linear bias
					weight_copy = m.bias.data.abs().clone().cpu().numpy()
					grad_copy = m.bias.grad.data.abs().clone().cpu().numpy()
					
					taylor = weight_copy*grad_copy  #

					arg_max = np.argsort(taylor)
					arg_max_rev = arg_max[::-1][:num_keep]
					thre = taylor[arg_max_rev[-1]]
					mask = np.zeros(weight_copy.shape[0])
					mask_R = np.ones(weight_copy.shape[0])
					mask[arg_max_rev.tolist()] = 1.0
					mask_R[arg_max_rev.tolist()] = 0.0
					mask_list_4d.append(mask)
					mask_list_R_4d.append(mask_R)
					threshold_list.append(thre)
					gradient_list.append(m.bias.grad.data.clone())
					weight_list.append(m.bias.data.clone())
					taylor_list.append(taylor)
			i += 1
		all_mask = []
		all_mask.append(mask_list_4d)
		all_mask.append(mask_list_R_4d)
		print('Got some lists: mask/maskR/threshold/gradient/weight/{}'.format(args.score))
		print('mask length: {} // threshold_list length:{} // gradient list: length {} // weight list: length {} // taylor_list: length {}'.
					 format(len(mask_list_4d), len(threshold_list), len(gradient_list), len(weight_list), len(taylor_list)))  # 33

		gradient_dict, threshold_dict, mask_dict, mask_R_dict, taylor_dict = self.convert_list_to_dict(gradient_list, threshold_list, all_mask, taylor_list)
		return all_mask, threshold_dict, mask_dict, mask_R_dict, taylor_dict


	def mask_frozen_weight(self, maskR):

		param_processed = OrderedDict([(k, None) for k in self.net.state_dict().keys()])

		for layer_name, param in self.net.state_dict().items():
				param_processed[layer_name] = Variable(torch.mul(param, maskR[layer_name]), requires_grad=True)
		self.net.load_state_dict(param_processed)


	def AND_twomasks(self, mask_dict_1, mask_dict_2, maskR_dict_1, maskR_dict_2):
		maskR_processed = OrderedDict([(k, None) for k in maskR_dict_1.keys()])
		mask_processed = OrderedDict([(k, None) for k in maskR_dict_1.keys()])
		for layer_name, mask in maskR_dict_1.items():
			maskR_processed[layer_name] = torch.mul(maskR_dict_1[layer_name], maskR_dict_2[layer_name])
			mask_processed[layer_name] =  torch.add(mask_dict_1[layer_name], mask_dict_2[layer_name])
		return mask_processed, maskR_processed

