import argparse
import torch
import os

def args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--method', type=str, default='mnist', help="name of method")
    parser.add_argument('--iid_level', type=int, default=4, help='number of data classes for local clients')
    parser.add_argument('--numclass', type=int, default=10, help="number of data classes in the first task")
    parser.add_argument('--increase_class', type=int, default=1, help="number of increasing data classes in each new task")
    parser.add_argument('--img_size', type=int, default=32, help="size of images")
    parser.add_argument('--device_id', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--batch_size', type=int, default=64, help='size of mini-batch')
    parser.add_argument('--task_size', type=int, default=2, help='number of data classes each task')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--memory_size', type=int, default=2000, help='size of exemplar memory')
    parser.add_argument('--epochs_local', type=int, default=20, help='local epochs of each global round')
    parser.add_argument('--learning_rate', type=float, default=2.0, help='learning rate')
    parser.add_argument('--num_clients', type=int, default=4, help='initial number of clients')
    parser.add_argument('--local_clients', type=int, default=4, help='number of selected clients each round')
    parser.add_argument('--epochs_global', type=int, default=50, help='total number of global rounds')
    parser.add_argument('--tasks_global', type=int, default=10, help='total epoch of each task')

    parser.add_argument('--clusting', type=str, default='fedavg', help='way of clusting')
    parser.add_argument('--temperature', type=float, default=5.0, help='total number of tasks')

    parser.add_argument('--local_rank', default=0, type=int, help='rank of distributed processes')
    parser.add_argument('--device_ids', type=str, default='0,1,2,3,4,5,6,7')
    args = parser.parse_args()
    return args