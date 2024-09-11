import torch.nn as nn
import torch
import copy
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
from myNetwork import *
from torch.utils.data import DataLoader
import random
from Fed_utils import *
from option import args_parser

args = args_parser()

class proxyServer:
    def __init__(self, device, learning_rate, numclass, feature_extractor, encode_model, test_transform):
        super(proxyServer, self).__init__()
        self.Iteration = 250
        self.learning_rate = learning_rate
        self.model = network(numclass, feature_extractor)
        self.encode_model = encode_model
        self.monitor_dataset = Proxy_Data(test_transform)  # 实例化
        self.new_set = []
        self.new_set_label = []
        self.numclass = 0
        self.device = device
        self.num_image = 20
        self.pool_grad = None
        self.best_model_1 = None
        self.best_model_2 = None
        self.best_perf = 0
        # self.monitor_loader = None

    def dataloader(self, pool_grad):
        self.pool_grad = pool_grad
        # 梯度池不为空
        if len(pool_grad) != 0:
            # 获得重构的样本图片的标签，用来评估旧模型
            if args.local_rank == 0:
                print('开始重构伪样本')
            self.reconstruction()
            # print('伪样本已重建好')
            # 拿到监控数据（重构的）监控数据是随机生成的伪图片经过重构
            self.monitor_dataset.getTestData(self.new_set, self.new_set_label)
            self.monitor_loader = DataLoader(dataset=self.monitor_dataset, shuffle=True, batch_size=1, drop_last=True)
            if args.local_rank == 0:
                print('self.monitor_loader已构建好')
            self.last_perf = 0
            self.best_model_1 = self.best_model_2

        cur_perf = self.monitor()
        if args.local_rank == 0:
            print(f'当前平均模型在伪样本上精度为{cur_perf}')
        # 如果新一轮数据的精度大于过去最好精度，更新最好精度，更新全局最佳旧模型
        if cur_perf >= self.best_perf:
            if args.local_rank == 0:
                print('当前模型精度更高，需更新全局最佳旧模型')
            self.best_perf = cur_perf
            self.best_model_2 = copy.deepcopy(self.model)

    def model_back(self):
        return [self.best_model_1, self.best_model_2]

    # 获得对监控数据的精度
    def monitor(self):
        self.model.eval()
        correct, total = 0, 0
        for step, (imgs, labels) in enumerate(self.monitor_loader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)  # 此时模型已被设置为完成联邦平均的全局模型
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct / total
        
        return accuracy

    def gradient2label(self):
        pool_label = []
        for w_single in self.pool_grad:
            # 观察最后一层的梯度符号（在[46，58]中提出）来获得其对应的真值标签（p-1，真实的梯度为负值）
            pred = torch.argmin(torch.sum(w_single[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)
            pool_label.append(pred.item())

        return pool_label

    # 重构样本
    def reconstruction(self):
        self.new_set, self.new_set_label = [], []

        tt = transforms.Compose([transforms.ToTensor()])
        tp = transforms.Compose([transforms.ToPILImage()])
        pool_label = self.gradient2label()  # 一维列表
        pool_label = np.array(pool_label)   # 一维标签的数组
        # print(pool_label)
        class_ratio = np.zeros((1, 100))

        for i in pool_label:  # 一般梯度池的图片类别可能会重复（有不同的客户）
            class_ratio[0, i] += 1

        for label_i in range(100):
            if class_ratio[0, label_i] > 0:
                # print(f'重构标签{label_i}的伪图片')
                num_augmentation = self.num_image
                augmentation = []

                # 获得指定标签在梯度池中的索引，梯度池是打乱的,
                grad_index = np.where(pool_label == label_i)  # where返回的是元组，元组中只有应该元素即array（坐标）

                for j in range(len(grad_index[0])):
                    # print('reconstruct_{}, {}-th'.format(label_i, j))
                    grad_truth_temp = self.pool_grad[grad_index[0][j]] # 获得同一标签所在不同位置的梯度

                    # 随机生成一个伪图片
                    dummy_data = torch.randn((1, 3, 32, 32)).to(self.device).requires_grad_(True)
                    label_pred = torch.Tensor([label_i]).long().to(self.device).requires_grad_(False)
                    # 针对该伪图片进行更新
                    # print('伪图片设备为：', dummy_data.device)

                    recon_model = copy.deepcopy(self.encode_model)
                    # print('recon_model设备为：', next(recon_model.parameters()).device)

                    # recon_model = recon_model.cuda(1)
                    # print(next(recon_model.parameters()).device)

                    optimizer = optim.SGD([dummy_data, ], lr=0.1)
                    criterion = nn.CrossEntropyLoss()

                    for iters in range(self.Iteration):

                        optimizer.zero_grad()
                        pred = recon_model(dummy_data)
                        dummy_loss = criterion(pred, label_pred)
                        # 输出值对输入变量求导
                        dummy_dy_dx = torch.autograd.grad(dummy_loss, recon_model.parameters(), create_graph=True)

                        grad_diff = 0    # 梯度            梯度
                        for gx, gy in zip(dummy_dy_dx, grad_truth_temp):
                            grad_diff += ((gx - gy) ** 2).sum()

                        grad_diff.backward()
                        optimizer.step()  # 对应原文中的公式8，9
                        current_loss = grad_diff.item()


                        # if iters == self.Iteration - 1:
                        #     print(current_loss)

                        if iters >= self.Iteration - self.num_image: # 保存最后更新的20张图片
                            # dummy_data_temp = np.asarray(tt(dummy_data.clone().squeeze(0).cpu()))
                            dummy_data_temp = np.asarray(tp(dummy_data.clone().squeeze(0).cpu()))
                            # print('dummy_data_temp:', dummy_data_temp.device, 'dummy_data.device:', dummy_data.device)
                            augmentation.append(dummy_data_temp)  # 同一标签的伪样本
                            # print('augmentation长度为：', len(augmentation)),最后都是为20张

                self.new_set.append(augmentation)
                self.new_set_label.append(label_i)

        # print('伪数据已构建好')


    