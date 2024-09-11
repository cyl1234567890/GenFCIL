import torch.nn as nn
import torch
import copy
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from myNetwork import *
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader
import random
from option import args_parser
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

args = args_parser()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def model_to_device(model, parallel, device, device_ids):
    if parallel:
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device_ids[args.local_rank]],
                                                    output_device=device_ids[args.local_rank],
                                                    find_unused_parameters=True)
    else:
        model = model.to(device)

    return model
def participant_exemplar_storing(clients, num, model_g, old_client, task_id, clients_index):
    for index in range(num):
        if index not in clients_index:

            clients[index].model = copy.deepcopy(model_g)

        if index not in clients_index:

            if index in old_client:
                clients[index].beforeTrain(index, task_id, 0)
            else:
                clients[index].beforeTrain(index, task_id, 1)
            #clients[index].update_new_set(index)

        clients[index].signal = False

def local_train(clients, index, model_g, task_id, oldmodel, model_old, ep_g, old_client, epg,z):
    if ep_g >=1:
        clients[index].model = copy.deepcopy(model_g)
    clients[index].epg = epg
    if index in old_client:
        available_labels=clients[index].beforeTrain(index,task_id, 0)
    else:
        available_labels=clients[index].beforeTrain(index,task_id, 1)
    #clients[index].update_new_set(index)

    if epg >=1:
        is_gen=epg//10+1
    else:
        is_gen=0
    feature = clients[index].train(ep_g, index, oldmodel, model_old, z,is_gen=is_gen)
    # 获得本地模型参数和梯度池
    if isinstance(clients[index].model, torch.nn.parallel.DistributedDataParallel):
        local_model = clients[index].model.fc.module.state_dict()
       # local_linear = clients[index].model.fc.module.state_dict()
    else:
        local_model = clients[index].model.state_dict()
       # local_linear = clients[index].model.fc.module.state_dict()
    return copy.deepcopy(local_model), copy.deepcopy(clients[index].model.fc), clients[index].model,feature,available_labels
    #return local_model,clients[index].model, available_labels

def Fed_Avg(models):
    w_avg = copy.deepcopy(models[0])
    for k in w_avg.keys():
        for i in range(1, len(models)):
            w_avg[k] += models[i][k]
        w_avg[k] = torch.div(w_avg[k], len(models))
    return w_avg
def Fed_DeltaPoolCls(linear_model, label_set,local_feature,pool="max",norm=-1):
    z_b0=[]
    for i in range(len(linear_model)):
        if norm > 0:
            normalized_weights = F.normalize(linear_model[i].weight, p=2, dim=1) # 权重进行L2范数归一化
            linear_model[i].weight.data = normalized_weights
            z_b0.append( linear_model[i](local_feature[i]))
        else:
            z_b0.append( linear_model[i](local_feature[i]))
    print(z_b0)
    z_b=[]
    z_total = []
    for _ in range(args.numclass):
        z_total.append(0)
    delta_z_b=0
    for i in range(len(linear_model)):
        for j in range(len(linear_model)):
            if j != i:
                delta_z_b += (linear_model[i](local_feature[j]))
        z_b.append( z_b0[i]+ delta_z_b /(len(linear_model)-1))   #z_b:里面列表数为客户数
        #z_b.append(z_b0[i])
    '''
    for j in range(args.numclass):
        for i in range(len(label_set) - 1):
            if j in label_set[i]:
                if z_total[j] ==0:
                    z_total[j]=z_b[i][j]
                else:
                    #z_overlap = torch.cat(
                    #[
                    #    z_total[j].unsqueeze(0),
                    #    z_b[i][j].unsqueeze(0)
                    #],
                    #dim=0
                    #)
                    #z_overlap = max( z_overlap).squeeze()
                    z_total[j]=z_overlap=max(z_total[j],z_b[i][j])
                    z_b[i][j] = z_overlap
    #z = z_total[0].unsqueeze(0)
    #for i in range(1, len(z_total)):
    #    z = torch.cat((z.cuda(), z_total[i].unsqueeze(0).cuda()))
            #z = torch.cat([z_overlap, z_b[i][:, self.n_overlap:], z_n[i+1][:, self.n_overlap:]], dim=1)
    #if route:
    #    route_out = torch.cat((z_b.max(dim=1)[0].unsqueeze(-1), z_n.max(dim=1)[0].unsqueeze(-1)),dim=1,)
    #    return z, route_out
    #for k in w_new.keys():
    #    if 'fc' in k:
    #        w_new[k] = z
    softmax = nn.Softmax(dim=0)
    for z_list in z_b:
        for i in range(len(z_list)):
            z_list[i] /= 10
        z_list = softmax(torch.tensor(z_list))
    '''
    return z_b

def model_global_eval(z,model_g, test_dataset, task_id, ep_g, task_size, device, device_ids,n):
    # model_to_device(model_g, False, device, device_ids)
    model_g.eval()
    test_dataset.getTestData([0, (task_id+1)])
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=128)
    correct, total = 0, 0
    true_labels = []
    predicted_labels = []
    for setp, (indexs, imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.cuda(device), labels.cuda(device)
        with torch.no_grad():
            outputs = model_g(imgs)
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        total += len(labels)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicts.cpu().numpy())

    if n ==1 and ep_g >=40:    # 生成混淆矩阵
        classes = list(range(args.numclass))
        cm = confusion_matrix(true_labels, predicted_labels, labels=classes)

    # 输出混淆矩阵和分类准确率
    #print('Confusion Matrix:')
    #print(cm)

        accuracy = 100 * correct / total

        plot_confusion_matrix(cm, classes,task_id, ep_g)
        plt.savefig(' Confusion Matrix of Task: {}, Round: {},Dataset: {}'.format(task_id, ep_g,args.dataset))
    else:
        accuracy = 100 * correct / total
    model_g.train()
    return accuracy

def plot_confusion_matrix(cm, classes, task_id, ep_g):
    # 绘制热力图
    plt.figure(figsize=(10, 10))
    heatmap=plt.imshow(cm, interpolation='nearest', cmap=plt.cm.magma, aspect='equal')
    #plt.title('Confusion Matrix of Task: {}, Round: {},Dataset: {}'.format(task_id, ep_g,args.dataset))
    colorbar = plt.colorbar(heatmap, shrink=0.8)
    colorbar.ax.tick_params(labelsize=16)
    plt.axis('off')
    #tick_marks = np.arange(len(classes))

    # 在热力图上显示数值
    #thresh = cm.max() / 2.
    #for i in range(cm.shape[0]):
    #    for j in range(cm.shape[1]):
    #        plt.text(j, i, cm[i, j],
    #                 horizontalalignment="center",
    #                 color="white" if cm[i, j] > thresh else "black")
    #plt.plot(range(cm.shape[0]), range(cm.shape[0]), color='yellow', linewidth=1)
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
    plt.show()
def select_users(num_clients, local_clients):
        return random.sample(range(num_clients), local_clients)