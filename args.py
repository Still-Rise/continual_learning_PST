
"""
Author: Xiaocong Du
Description: 
Title: Single-Net Continual Learning with Progressive Segmented Training (PST)
"""

import argparse

parser = argparse.ArgumentParser(description='Single-Net Continual Learning with Progressive Segmented Training (PST)')
parser.add_argument('--gpu', type=str, default = '0', help='GPU')
parser.add_argument('--seed', type=int, default = 33, help='random seed')
parser.add_argument('--resume', type=bool, default = False, help='resume checkpoint')

parser.add_argument('--dataset', default = 'cifar100', type=str, choices=['cifar100'])
parser.add_argument('--classes_per_task', type=int, default = 20, choices=[5, 10, 20, 50], help='classes per task')
parser.add_argument('--num_classes', type=int, default = 120, help='final FC size')
parser.add_argument('--shuffle', type=bool, default = True, help='dataset shuffle')

parser.add_argument('--num_epoch', type=int, default = 100, help='training epochs')
parser.add_argument('--NA_C0', type=int, default = 30, help='size of first channel in resnet')

parser.add_argument('--batch_size', type=int, default = 128, help='batch size')
parser.add_argument('--weight_decay', default = 5E-4, type=float, help='weight decay')

parser.add_argument('--lr', default = 0.1, type=float, help='learning rate')
parser.add_argument('--lr_step_size', default = 40, type=int, help='learning rate decay step')
parser.add_argument('--lr_gamma', default = 0.1, type=float, help='learning rate decay rate')

parser.add_argument('--score', type=str, default = 'grad_w', choices=['grad_w'], help='importance score')

parser.add_argument('--total_memory_size', type=int, default = 2000, help='memory size')

# parser.add_argument('--first_task_number', type=int, default = 5, choices=[1, 3, 5, 7, 9], help='for asymmetric experiment, X+1 denotes 10+10, 30+10, 50+10 ...')

