"""
Author: anonymous
Description: 
Supplementary material for NeurIPS2019 paper submission
Title: Single-Net Continual Learning with Progressive Segmented Training (PST)
Submisstion ID: 2994
"""

import os
import pickle
import sys
import scipy.io as scio
import continualNN
from load_cifar import *
import matplotlib.pyplot as plt

from args import parser
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

print("*************************************************************************************************")
print("                                         PST_main.py                                             ")
print("*************************************************************************************************")
print("args = %s", args)

method = continualNN.ContinualNN()
method.initial_single_network(init_weights = True )
method.initialization(args.lr, args.lr_step_size, args.weight_decay)

task_list, total_task = method.create_task()
print('Task list: ', task_list)

num_epoch0 = args.num_epoch
num_epoch1 = int(args.num_epoch * 0.3)
num_epoch2 = int(args.num_epoch * 0.6)
num_epoch3 = int(args.num_epoch * 0.1)

train_acc = []  
test_acc_0 = []  
test_acc_current = [] 
test_acc_mix = [] 
test_multihead_0 = [] 
test_multihead_current = [] 
test_task_accu = []  # At the end of each task, best overall test accuracy. Length = number of tasks
test_acc_0_end = []  # At the end of each task, the accuracy of task 0. Length = number of tasks
NME_accu_mix = []
NME_accu_0 =[]
NME_accu_current = []

beta = args.classes_per_task / 100
print("==================================  Train task 0 ==========================================")
"""Test data from the first task"""
train_0, test_0 = get_dataset_cifar(task_list[0], 0*args.classes_per_task)
for batch_idx, (data, target) in enumerate(train_0):
	print('task 0\n', np.unique(target))
	break

best_acc_0 = 0.0
for epoch in range(num_epoch0):
	train_acc.append(method.train(epoch, train_0))
	test_acc_0.append(method.test(test_0))
	test_acc_current.append(np.zeros(1))
	test_acc_mix.append(np.zeros(1))
	test_multihead_0.append(np.zeros(1))
	test_multihead_current.append(np.zeros(1))

	if test_acc_0[-1] > best_acc_0:
		best_acc_0 = test_acc_0[-1]
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head This training on T0 testing accu is : {:.4f}'.format(method.test(test_0)))
	print('train_acc {0:.4f}\n\n\n'.format(train_acc[-1]))
method.save_model(0, 0)

current_mask_list, current_threshold_dict, mask_dict_pre, maskR_dict_pre, current_taylor_dict = method.sensitivity_rank_taylor_filter(beta)
with open('../mask_library/mask_task{}_threshold{}_acc{:.4f}.pickle'.format(0, beta, best_acc_0), "wb") as f:
	pickle.dump((current_mask_list, current_threshold_dict, mask_dict_pre, maskR_dict_pre, current_taylor_dict), f)


print("================================== task 0: Clear cut 0 of task 0 ==========================================")
method.initial_single_network(init_weights = True)
method.initialization(args.lr, int(num_epoch2*0.7), args.weight_decay)
method.load_model_random_initial(maskR_dict_pre, mask_dict_pre, model_id=0, task_id= 0)  # initialize top important beta0 weights and retrain from scratch

for epoch in range(num_epoch2):
	train_acc.append(method.train_with_frozen_filter(epoch, train_0, maskR_dict_pre, mask_dict_pre))
	test_acc_0.append(method.test(test_0))
	test_acc_current.append(np.zeros(1))
	test_acc_mix.append(np.zeros(1))
	test_multihead_0.append(np.zeros(1))
	test_multihead_current.append(np.zeros(1))
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head This training on T0 testing accu is : {:.4f}'.format(method.test(test_0)))
	print('Cut T0 train_acc {0:.4f}\n\n\n'.format(train_acc[-1]))
	if test_acc_0[-1] > best_acc_0:
		best_acc_0 = test_acc_0[-1]
test_task_accu.append(best_acc_0)
test_acc_0_end.append(best_acc_0)

torch.save(method.net.state_dict(), '../results/model_afterT{0}_Accu{1:.4f}.pt'.format(0, best_acc_0))


for task_id in range(1, total_task):
	print("================================== 1. Current Task is {} : Prepare dataset ==========================================".format(task_id))

	"""Current trainset, e.g. task id = 2, the 3rd task"""
	# task_id = 1
	train_current, test_current = get_dataset_cifar(task_list[task_id], task_id*args.classes_per_task)
	for batch_idx, (data, target) in enumerate(train_current):
		print('train_current re-assigned label: \n', np.unique(target))
		break

	"""Balance memory: same amounts of images from previous tasks and current task"""
	memory_each_task = int(args.total_memory_size / task_id) # The previous tasks shares the memory
	alltask_list = []
	alltask_memory = []
	alltask_single_list = []
	for i in range(task_id+1):
		alltask_list.append(task_list[i]) 
		alltask_memory.append(memory_each_task)
		alltask_single_list += task_list[i]
	train_bm, _ = get_partial_dataset_cifar(0, alltask_list, num_images = alltask_memory)
	for batch_idx, (data, target) in enumerate(train_bm):
		print('train_bm (balanced memory) re-assigned label: \n', np.unique(target))
		break

	"""Test data from all the tasks"""
	_, test_mix_full = get_dataset_cifar(alltask_single_list, 0)
	for batch_idx, (data, target) in enumerate(test_mix_full):
		print('test_mix_full (all test data till now) re-assigned label: \n', np.unique(target))
		break


	print("=============================== 2. Current Task is {} : Memory-assisted balancing ==================================".format(task_id))
	method.initialization(args.lr, int(num_epoch1*0.7), args.weight_decay)
	for epoch in range(num_epoch1):
		train_acc.append(method.train_with_frozen_filter(epoch, train_bm, mask_dict_pre, maskR_dict_pre))
		test_acc_0.append(method.test(test_0))
		test_acc_current.append(method.test(test_current))
		test_acc_mix.append(method.test(test_mix_full))
		test_multihead_0.append(method.test_multihead(0, test_0))
		test_multihead_current.append(method.test_multihead(task_id, test_current))
		print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head T0 testing accu is : {:.4f}'.format( test_acc_0[-1]))
		print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Multi-headed T0 testing accu is : {:.4f}'.format( test_multihead_0[-1]))
		print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head current testing accu is : {:.4f}'.format( test_acc_current[-1]))
		print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Multi-headed current testing accu is : {:.4f}'.format( test_multihead_current[-1]))
		print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head mixed all tasks testing accu is : {:.4f}'.format( test_acc_mix[-1]))
		print('train_acc {0:.4f} \n\n\n'.format(train_acc[-1]))

	print("Current Task is {} : Train task 1 w/ injecting memory iteratively ====================".format(task_id))
	method.initialization(args.lr, args.lr_step_size, args.weight_decay)
	best_acc_mix = 0.0
	for epoch in range(num_epoch2):
		if epoch % 3 == 0 or epoch % 5 == 0 or epoch > num_epoch2 - 3:
			train_acc.append(method.train_with_frozen_filter(epoch, train_bm, mask_dict_pre, maskR_dict_pre))
		else:
			train_acc.append(method.train_with_frozen_filter(epoch, train_current, mask_dict_pre, maskR_dict_pre))

		test_acc_0.append(method.test(test_0))
		test_acc_current.append(method.test(test_current))
		test_acc_mix.append(method.test(test_mix_full))
		test_multihead_0.append(method.test_multihead(0, test_0))
		test_multihead_current.append(method.test_multihead(task_id, test_current))
		if test_acc_mix[-1] > best_acc_mix:
			best_acc_mix = test_acc_mix[-1]
		print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head T0 testing accu is : {:.4f}'.format( test_acc_0[-1]))
		print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Multi-headed T0 testing accu is : {:.4f}'.format( test_multihead_0[-1]))
		print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head current testing accu is : {:.4f}'.format( test_acc_current[-1]))
		print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Multi-headed current testing accu is : {:.4f}'.format( test_multihead_current[-1]))
		print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head mixed all tasks testing accu is : {:.4f}'.format( test_acc_mix[-1]))
		print('train_acc {0:.4f} \n\n\n'.format(train_acc[-1]))


	print("Current Task is {} : Finetune FC with balanced memory =================================".format(task_id))
	method.initialization(args.lr*0.5, int(num_epoch3*0.7), args.weight_decay)

	for epoch in range(num_epoch3):
		train_acc.append(method.train_fc(epoch, train_bm))

		test_acc_0.append(method.test(test_0))
		test_acc_current.append(method.test(test_current))
		test_acc_mix.append(method.test(test_mix_full))
		test_multihead_0.append(method.test_multihead(0, test_0))
		test_multihead_current.append(method.test_multihead(task_id, test_current))
		if test_acc_mix[-1] > best_acc_mix:
			best_acc_mix = test_acc_mix[-1]
		print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head T0 testing accu is : {:.4f}'.format( test_acc_0[-1]))
		print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Multi-headed T0 testing accu is : {:.4f}'.format( test_multihead_0[-1]))
		print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head current testing accu is : {:.4f}'.format( test_acc_current[-1]))
		print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Multi-headed current testing accu is : {:.4f}'.format( test_multihead_current[-1]))
		print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head mixed all tasks testing accu is : {:.4f}'.format( test_acc_mix[-1]))
		print('train_acc {0:.4f} \n\n\n'.format(train_acc[-1]))


	test_task_accu.append(best_acc_mix)
	test_acc_0_end.append(test_acc_0[-1])

	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> At the end of task {}, T0 accu is {:.4f}'.format(task_id, test_acc_0[-1]))
	method.save_model(0, task_id)
	torch.save(method.net.state_dict(), '../results/model_afterT{0}_Accu{1:.4f}.pt'.format(task_id, best_acc_mix))


	if task_id != total_task-1:
		print("===================================== 3.  Current Task is {} : importance sampling ====================================".format(task_id))
		method.mask_frozen_weight(maskR_dict_pre)

		current_mask_list, current_threshold_dict, mask_dict_current, maskR_dict_current, current_taylor_dict = method.sensitivity_rank_taylor_filter(beta)
		with open('../mask_library/mask_task{}_threshold{}_acc{:.4f}.pickle'.format(task_id, beta, best_acc_mix), "wb") as f:
			pickle.dump((current_mask_list, current_threshold_dict, mask_dict_current, maskR_dict_current, current_taylor_dict, mask_dict_pre, maskR_dict_pre), f)


		print("===================================== 4. Current Task is {} : model segmentation ==========================================".format(task_id))
		method.initial_single_network(init_weights=True)
		method.initialization(args.lr, int(num_epoch2*0.7), args.weight_decay)
		method.load_model_random_initial(maskR_dict_current, mask_dict_current, model_id=0, task_id = task_id)  # initialize top important beta0 weights and retrain from scratch

		for epoch in range(num_epoch2):
			if epoch % 3 == 0 or epoch % 5 == 0 or  epoch > num_epoch2 - 3:
				train_acc.append(method.train_with_frozen_filter(epoch, train_bm, maskR_dict_current, mask_dict_current))
			else:
				train_acc.append(method.train_with_frozen_filter(epoch, train_current, maskR_dict_current, mask_dict_current))

			test_acc_0.append(method.test(test_0))
			test_acc_current.append(method.test(test_current))
			test_acc_mix.append(method.test(test_mix_full))
			test_multihead_0.append(method.test_multihead(0, test_0))
			test_multihead_current.append(method.test_multihead(task_id, test_current))
			print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head T0 testing accu is : {:.4f}'.format( test_acc_0[-1]))
			print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Multi-headed T0 testing accu is : {:.4f}'.format( test_multihead_0[-1]))
			print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head current testing accu is : {:.4f}'.format( test_acc_current[-1]))
			print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Multi-headed current testing accu is : {:.4f}'.format( test_multihead_current[-1]))
			print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head mixed all tasks testing accu is : {:.4f}'.format( test_acc_mix[-1]))

		print("Current Task is {} : Combine masks  ==========================================".format(task_id))
		mask_dict_pre, maskR_dict_pre = method.AND_twomasks(mask_dict_pre, mask_dict_current, maskR_dict_pre, maskR_dict_current)


		current_mask_list_afterMS, current_threshold_dict_afterMS, mask_dict_current_afterMS, maskR_dict_current_afterMS, current_taylor_dict_afterMS = method.sensitivity_rank_taylor_filter(beta)
		with open('../mask_library/mask_task{}_afterMS_threshold{}_acc{:.4f}.pickle'.format(task_id, beta, best_acc_mix), "wb") as f:
			pickle.dump((current_mask_list_afterMS, current_threshold_dict_afterMS, mask_dict_current_afterMS, maskR_dict_current_afterMS, current_taylor_dict_afterMS, mask_dict_pre, maskR_dict_pre), f)

## RESULTS DOCUMENTATION
print("====================== Document results ======================")

plt.figure()
x = np.linspace(0, len(test_task_accu), num = len(test_task_accu))
plt.xlim(0, 100/args.classes_per_task)
plt.xlabel('Task ID')
plt.ylabel('Accuracy')
plt.plot(x, test_task_accu , 'g-o', alpha=1.0, label = 'our method')
plt.yticks(np.arange(0, 1.0, step=0.1))
plt.legend(loc='best')
plt.title('Incrementally learning {} classes at a time'.format(args.classes_per_task))
plt.savefig('../results/incremental_curve_Task{}_{:.4f}.png'.format(task_id, best_acc_mix))
plt.show()


