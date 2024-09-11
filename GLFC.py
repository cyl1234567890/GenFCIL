import torch
import torch.nn as nn

from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
from option import args_parser

from myNetwork import *
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import random
import time
from Fed_utils import *


import itertools


args = args_parser()

def get_one_hot(target, num_class, device):
    target = target.to(device)
    one_hot = torch.zeros(target.shape[0], num_class).to(device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot



def entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

class GLFC_model:

    def __init__(self, client_id, numclass, feature_extractor, gen, batch_size, task_size, memory_size, epochs,
                 learning_rate, train_set, device, oldmodel, device_ids):

        super(GLFC_model, self).__init__()
        self.client_id = client_id
        self.device_ids = device_ids
        self.epg = None
        self.epochs = epochs  # 默认20
        self.learning_rate = learning_rate
        self.model = network(numclass, feature_extractor)  # 获得模型
        self.feature_extractor = feature_extractor

        self.index = None
        self.exemplar_set = []
        self.class_mean_set = []
        self.numclass = 0
        self.learned_numclass = 0
        self.learned_classes = []
        self.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.old_model = None
        self.train_dataset = train_set
        self.start = True
        self.signal = False

        self.batchsize = batch_size
        self.memory_size = memory_size
        self.task_size = task_size

        self.train_loader = None
        self.current_class = None
        self.last_class = None
        self.task_id_old = -1  # -1表示没有任务（初始）第一轮任务id为0
        self.device = device
        self.last_entropy = 0
        #生成器
        self.generative_alpha = 1
        self.generative_beta = 1
        self.generative_model = gen
        self.latent_layer_idx = -1
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")
        self.available_labels = []
        self.oldmodel = oldmodel
        self.old_data=0
        self.old_label=0

        self.lendata= 0
        self.new_data=[]
        self.old_feature_extractor = None
        self.old_label=[]
        self.z_old=torch.zeros(10)
    def beforeTrain(self,index, task_id_new, group):
        if task_id_new != self.task_id_old:
            self.signal = True
            #self.task_id_old = task_id_new
            self.numclass = args.numclass
            if group != 0:
                if self.current_class != None:
                    self.last_class = self.current_class
                self.current_class = random.sample([x for x in range(0, args.numclass)], args.iid_level)
                #self.current_class = random.sample([x for x in range(2, self.numclass)], 2)
                #self.current_class.extend([0, 1])

            else:
                self.last_class = None
        self.train_loader, labels = self._get_train_and_test_dataloader(self.client_id,  task_id_new,self.current_class, False)
        return labels


    def update_new_set(self,index):
        self.model.eval()
        if self.signal and (self.last_class != None):
            self.learned_numclass += len(self.last_class)
            self.learned_classes += self.last_class
            m = int(self.memory_size / self.learned_numclass)
            self._reduce_exemplar_sets(m)
            for i in self.last_class:
                images = self.train_dataset.get_image_class(i)
                self._construct_exemplar_set(images, m)
        self.model.train()
        self.train_loader, labels = self._get_train_and_test_dataloader(self.client_id, task_id_new, self.current_class, True)


    def _get_train_and_test_dataloader(self,index,task_id_new, train_classes, mix):
        self.available_labels = []
        if mix:
            if len(self.exemplar_set)>0 and len(self.learned_classes)>0:

                label=self.train_dataset.getTrainData(index, task_id_new,train_classes, self.exemplar_set, self.learned_classes)
        else:
            label = self.train_dataset.getTrainData(index, task_id_new, train_classes, [], [])
            self.task_id_old = task_id_new

        #for i in label:
        for i in label:
            if i[0] not in self.available_labels:
                self.available_labels.extend([i[0]])
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize,
                                  num_workers=8,
                                  pin_memory=True)
        return train_loader, self.available_labels


    def train(self, ep_g, index,oldmodel,model_old,z, is_gen =False):
        self.epg = ep_g
        self.index = index
        self.oldmodel = oldmodel
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)

        self.generative_model.eval()
        TEACHER_LOSS, DIST_LOSS, LATENT_LOSS = 0, 0, 0
        if model_old[1] != None:
            if self.signal:
                self.old_model = model_old[1]
            else:
                self.old_model = model_old[0]
        else:
            if self.signal:
                self.old_model = model_old[0]
        if self.old_model != None:
            self.old_model = model_to_device(self.old_model, False, self.device, self.device_ids)
            self.old_model.eval()
        time_start = time.time()

        for epoch in range(args.epochs_local):
            running_loss = 0

            if args.dataset == 'mnist' or args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'svhn':
                if (epoch + ep_g * 20) % 200 == 100:
                    if self.numclass == self.task_size:
                        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 5, weight_decay=0.00001)
                    else:
                        for p in opt.param_groups:
                            p['lr'] = self.learning_rate / 5
                elif (epoch + ep_g * 20) % 200 == 150:
                    if self.numclass > self.task_size:
                        for p in opt.param_groups:
                            p['lr'] = self.learning_rate / 25
                    else:
                        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 25, weight_decay=0.00001)
                elif (epoch + ep_g * 20) % 200 == 180:
                    if self.numclass == self.task_size:
                        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125, weight_decay=0.00001)
                    else:
                        for p in opt.param_groups:
                            p['lr'] = self.learning_rate / 125

            '''
            if self.numclass == self.task_size:
                opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / (ep_g/5 +2), weight_decay=0.00001)
            else:
                for p in opt.param_groups:
                    p['lr'] = self.learning_rate / (ep_g/5 +2)
            '''
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(self.device), target.to(self.device)
                loss_value,feature,output = self._compute_loss(indexs, images, target,z)
                if args.dataset == 'cifar10' or args.dataset == 'mnist' :
                    #generative_alpha = 0.3
                    generative_alpha = 0.3
                if args.dataset == 'svhn':
                    #generative_alpha = 0.3
                    generative_alpha = 0.3
                if args.dataset == 'cifar100':
                    generative_alpha = 0.5
                    #generative_alpha = 0.1
                generative_beta =0.1
                loss_proximal = 0
                if is_gen > 0:
                    gen_output = self.generative_model( target, latent_layer_idx=self.latent_layer_idx)['output']
                    logit_given_gen = self.model(gen_output, index=-1)
                    target_p = F.softmax(logit_given_gen, dim=1).clone().detach()
                    user_latent_loss = generative_beta * self.ensemble_loss(F.log_softmax(output, dim=1), target_p)# target:只有自己的六个类
                    if args.dataset =='cifar100':
                        sampled_y = np.random.choice([x for x in range(0, self.numclass) if x not in self.current_class],64)
                    else:

                        sampled_y = np.random.choice([x for x in range(0, self.numclass)],64)
                    sampled_y = torch.tensor(sampled_y)
                    gen_result = self.generative_model(sampled_y, latent_layer_idx=self.latent_layer_idx)['output']
                    output = self.model(gen_result, index=-1)
                    teacher_loss = generative_alpha * torch.mean(self.generative_model.crossentropy_loss(F.log_softmax(output, dim=1).cuda(), sampled_y.cuda()))#sampled_y：自己没有的类
                    gen_ratio = 1

                    TEACHER_LOSS += teacher_loss
                    LATENT_LOSS += user_latent_loss
                    opt.zero_grad()
                    loss = loss_value + (gen_ratio * teacher_loss + user_latent_loss)
                    loss.backward()
                    opt.step()
                    running_loss += loss.item()
                else:
                    opt.zero_grad()
                    loss_value =loss_value + 0.5 * loss_proximal
                    loss_value.backward()
                    opt.step()
                    running_loss += loss_value.item()
        time_end = time.time()  # 结束计时
        t = time_end - time_start
        if args.local_rank == 0:
            print(
                f'客户{self.index}在第{self.epg}轮全局通信，第{self.task_id_old}轮任务中训练完成, 时间为{t}s ,generative_beta:{generative_beta},  generative_alpha:{ generative_alpha}  ,learn: 64',
                'LR: {:0.6f}'.format(opt.param_groups[0]['lr'] ))
        return feature

    def _compute_loss(self, indexs, imgs, label,z):
        if self.epg>=0:
            self.z = z[self.index].clone().detach().to(self.device)
        if self.epg>=10:
            self.z=self.logit_scores(self.last_class,self.z,self.z_old)
        if self.epg % 10==0:
        #z_clone.requires_grad = True
            self.z_old=self.z

        l=0
        #variance = torch.var(z_clone)
        output = self.model(imgs).cuda()

        feature = self.feature_extractor(imgs)
        target = get_one_hot(label, self.numclass, self.device)
        output, target = output.to(self.device), target.to(self.device)
        #if self.old_model == None:
        loss_cur = torch.mean(F.binary_cross_entropy_with_logits(output, target, reduction='none'))
        if self.epg>=10:
            l= torch.mean(F.binary_cross_entropy_with_logits(self.z *output, target, reduction='none'))
        return loss_cur+0.1*l, feature, output
        #return loss_cur +  variance.cuda() , feature, output
    '''
        else:
            loss_cur = torch.mean(F.cross_entropy(output, target, reduction='none'))
            distill_target = target.clone()
            with torch.no_grad():
                old_target = torch.sigmoid(self.old_model(imgs))

            old_task_size = old_target.shape[1]
            distill_target[..., :old_task_size] = old_target
            soft_loss = nn.KLDivLoss(reduction='batchmean')
            loss_old = soft_loss(torch.log(torch.sigmoid(output / args.temperature)), distill_target / args.temperature)
            return 0.5 * (loss_cur) + 0.5 * loss_old, output
        '''
    def logit_scores(self,old_label,z_total,z_old):
        for j in (self.current_class):
                if j in old_label:
                    z_total[j] = max(z_total[j], z_old[j])
                    #z_b[i][j] = z_overlap
            # z = z_total[0].unsqueeze(0)
            # for i in range(1, len(z_total)):
            #    z = torch.cat((z.cuda(), z_total[i].unsqueeze(0).cuda()))
            # z = torch.cat([z_overlap, z_b[i][:, self.n_overlap:], z_n[i+1][:, self.n_overlap:]], dim=1)
            # if route:
            #    route_out = torch.cat((z_b.max(dim=1)[0].unsqueeze(-1), z_n.max(dim=1)[0].unsqueeze(-1)),dim=1,)
            #    return z, route_out
            # for k in w_new.keys():
            #    if 'fc' in k:
            #        w_new[k] = z
        softmax = nn.Softmax(dim=0)
        z_total /= 10
        z_total = softmax(torch.tensor(z_total))
        self.z_old=z_total
        return z_total
    def _compute_old_loss(self, index,imgs, label):
        output = self.oldmodel(imgs)
        target = get_one_hot(label, self.numclass, self.device)
        output, target = output.to(self.device), target.to(self.device)
        loss_cur = torch.mean(F.binary_cross_entropy_with_logits(output, target, reduction='none'))  # 原文中的Lgc
        return loss_cur, output


    def _construct_exemplar_set(self, images, m):
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, 512))
        for i in range(m):
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])
        self.exemplar_set.append(exemplar)

    def _construct_current_set(self, images, m):
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, 512))
        for i in range(m):
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])
        self.new_data.append(exemplar)
    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
    def Image_transform(self, images, transform):
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data


    def compute_class_mean(self, images, transform):
        x = self.Image_transform(images, transform).to(self.device)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            feature_extractor_output = F.normalize(self.model.module.feature_extractor(x).detach()).cpu().numpy()
        else:
            feature_extractor_output = F.normalize(self.model.feature_extractor(x).detach()).cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output


    def compute_exemplar_class_mean(self):
        self.class_mean_set = []
        for index in range(len(self.exemplar_set)):
            exemplar = self.exemplar_set[index]
            class_mean, _ = self.compute_class_mean(exemplar, self.transform)
            class_mean_, _ = self.compute_class_mean(exemplar, self.classify_transform)  # 有问题
            class_mean = (class_mean / np.linalg.norm(class_mean) + class_mean_ / np.linalg.norm(class_mean_)) / 2
            self.class_mean_set.append(class_mean)


    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr = max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

class OldData(Dataset):
    def __init__(self, data,label,transform):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if len(self.data) != 0:
            return self.getoldTrainItem(index)
        return self.data[idx], self.label[idx]
    def getOldData(self, exemplar_set, exemplar_label_set):
        datas, labels = [], []
        if len(exemplar_set) != 0 and len(exemplar_label_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            length = len(datas[0])
            labels = [np.full((length), label) for label in exemplar_label_set]

        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label
    def getoldTrainItem(self, index):
        img, target = Image.fromarray(self.data[index]), self.label[index]

        if self.transform:
            img = self.transform(img)
        return index, img, target



