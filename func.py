#from Fed import average
from Fed import *
import logging
import torch
import math
import numpy as np
from numpy.random import choice
import torch.nn.functional as F
import pickle
import random
import heapq
from numpy import linalg as LA
from collections import Counter
import copy

logger = logging.getLogger("main_fed")
logger.setLevel(level=logging.DEBUG)

def result_avg(result):

    result_list = list(result.values())

    sums = Counter()
    counters = Counter()
    for itemset in result_list:
        sums.update(itemset)
        counters.update(itemset.keys())

    result = {x: float(sums[x])/counters[x] for x in sums.keys()}

    return result


def average(grad_all):
    value_list = list(grad_all.values())

    w_avg = copy.deepcopy(value_list[0])
    # print(type(w_avg))
    for i in range(1, len(value_list)):
        w_avg += value_list[i]
    return w_avg / len(value_list)


def FedAvg(w, idxs_users):
    # models = list(w.values())
    for index, value in enumerate(idxs_users):
        w_avg = w[0]
        for k in w_avg.keys():
            for i in range(1, len(idxs_users)):
                w_avg[k] += w[index][k]
            w_avg[k] = torch.div(w_avg[k], len(idxs_users))
    return w_avg
 

def subset(letter_ind, n):
    name_alphabet = list(map(chr, range(letter_ind, letter_ind+n)))
    return name_alphabet

## save variable
def save_obj(obj1, obj2, name):  
    pkl_path = "./review_result/niid/test/"
    with open(pkl_path + name + ".pkl", 'wb') as f:
        pickle.dump([obj1, obj2], f, pickle.HIGHEST_PROTOCOL)

def save_obj_more(obj1, obj2, obj3, name):  
    pkl_path = "./review_result/niid/test/"
    with open(pkl_path + name + ".pkl", 'wb') as f:
        pickle.dump([obj1, obj2, obj3], f, pickle.HIGHEST_PROTOCOL)

# def save_obj_more(obj1, obj2, obj3,obj4,obj5, name):  
#     pkl_path = "./review_result/iid/"
#     with open(pkl_path + name + ".pkl", 'wb') as f:
#         pickle.dump([obj1, obj2, obj3, obj4, obj5], f, pickle.HIGHEST_PROTOCOL)

def save_act_node(obj1, name):  
    pkl_path = "./review/"
    with open(pkl_path + name + ".pkl", 'wb') as f:
        pickle.dump([obj1], f, pickle.HIGHEST_PROTOCOL)

## load variable
def load_obj(name):
    pkl_path = "./result/test/"
    with open(pkl_path + name + ".pkl", 'rb') as f:
        return pickle.load(f)



def model_ini(traced_model):
    initial_model = {}
    params = traced_model.named_parameters()
    for name1, param1 in params:

        initial_model[name1] = param1.data

    return initial_model


def dot(K,L):
    temp = [i[0] * i[1] for i in zip(K, L)]
    ratio = temp.count(1) / len(temp)

    return round(ratio,2)
    
def dot_sum(K, L):

    return round(sum(i[0] * i[1] for i in zip(K, L)),2)#具体实现过程是将两个向量按元素对应相乘，然后对所有乘积结果求和。内积结果保留小数点后两位，使用了 round() 函数进行四舍五入。



def node_deleting(expect_list, expect_value, worker_ind, grads):#worker_ind：idxs_users

    # expect_list.pop("all")
    for i in range(len(worker_ind)):

        worker_ind_del  = [n for n in worker_ind if n != worker_ind[i]]   #除开i以外的剩余值赋给worker_ind_del
        grad_del = grads.copy()
        grad_del.pop(worker_ind[i])#除开i以外的剩余梯度值赋给avg_grad_del
        avg_grad_del = average(grad_del)
        grad_del['avg_grad'] = avg_grad_del
        expect_value_del = get_relation(grad_del, worker_ind_del)
        expect_list[worker_ind[i]] = expect_value_del
    expect_list['all'] =  expect_value

    return expect_list

  
def Diff(li1, li2):
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))


def get_gradient(args, pre, now, lr):
 
    grad = np.subtract( model_convert(pre), model_convert(now)) 
   
    return grad / (3000 * args.epochs_local * lr / args.batch_size)

def get_relation(avg_grad, idxs_users):

    innnr_value = {}
    for i in range(len(idxs_users)):
        
        innnr_value[idxs_users[i]] = dot_sum(avg_grad[idxs_users[i]], avg_grad['avg_grad']) #将当前用户与所有用户的平均梯度进行内积运算，并将结果存储在 innnr_value 字典中，键为当前用户的索引，值为内积结果。

    return round(sum(list(innnr_value.values())), 3)

def model_convert(model_grad):#ini 列表中存储了字典 model_grad 中所有键对应值的展平结果。

    ini = []
    for name in model_grad.keys():
     
        ini = ini + torch.flatten(model_grad[name]).tolist() 

    return ini


def probabilistic_selection(node_prob, node_count, act_indx, part_node_after, labeled, alpha):
    #logger = logging.getLogger("main_fed")
   # logger.setLevel(level=logging.DEBUG)

    remove_list = Diff(act_indx, part_node_after)
    print(remove_list)
    for i in remove_list:
        node_count[i][2] += 1

    # rest_nodes = Diff(list(node_prob.keys()), remove_list)

    rest_nodes = Diff(list(node_prob.keys()), labeled)
    beta = 0.7
    weight = 0
 
        
    ratio = {}
    for i in labeled:
        ratio[i] = node_count[i][1]/ node_count[i][0]
        
    for i in labeled:
        prob_change =  node_prob[i] * min( (ratio[i] + beta)**alpha, 1)
        logger.info(" node %s, rate_%s, change dis %s", i, node_count[i][1]/ node_count[i][0], min( (ratio[i] + beta)**alpha, 1) )
        weight += prob_change
        node_prob[i] =  node_prob[i] - prob_change
 
    for i in rest_nodes:
            node_prob[i] = node_prob[i] + weight / (len(rest_nodes))


    get_node = choice(list(node_prob.keys()), 10, replace=False, p=list(node_prob.values()))

 
    # print(node_explor)
    return get_node.tolist(), node_prob, node_count


def get_norm(graident):

    vari_norm = {}
    norm_vari = {}
    
    node_indx = list(graident.keys())
    node_indx.remove('avg_grad')

    for idx in list(graident.keys()):
        vari_norm[idx] = LA.norm(graident[idx])
        
    vari_norm = (sum(vari_norm.values()) - vari_norm['avg_grad']) / (len(vari_norm)-1) / vari_norm['avg_grad']

    for idx in node_indx:
        norm_vari[idx] = LA.norm( (graident[idx] - graident['avg_grad']) )
        
   

    return vari_norm, sum(norm_vari.values())/len(norm_vari)