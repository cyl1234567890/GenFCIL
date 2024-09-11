from torchvision.datasets import CIFAR100
import numpy as np
import torch
from PIL import Image
import random
import torch.nn.init as init
from option import args_parser

args = args_parser()
class iCIFAR100(CIFAR100):
    def __init__(self,root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=False):
        super(iCIFAR100,self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)

        self.target_test_transform=target_test_transform
        self.test_transform=test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []
        self.task_id=-1
        self.last=[]
    def concatenate(self, datas, labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1, len(datas)):
            con_data=np.concatenate((con_data, datas[i]), axis=0)  # 首位拼接，即变成一维了
            con_label=np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, classes):
        datas,labels=[],[]
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TestData, self.TestLabels=self.concatenate(datas, labels)

    def getTrainData(self, index, task_id_new, classes, exemplar_set, exemplar_label_set):
        # 获得存储器里面的数据和标签
        datas, labels = [], []
        if len(exemplar_set) != 0 and len(exemplar_label_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            length = len(datas[0])
            labels = [np.full((length), label) for label in exemplar_label_set]
        #if self.task_id !=task_id_new  and index not in self.last:
            #self.last.append(index)
        for label in classes:
            data_ = self.data[np.array(self.targets) == label]
            #indices = np.random.choice(len(data_), size=100, replace=False)
            #data = data_[indices]
            if task_id_new < 5:
                data = data_[index * 15 + task_id_new * 100 : index * 15 + 50 + task_id_new * 100]
            else:

                task_id_new = task_id_new - 5
                if index ==0:
                    indice1=data_[task_id_new * 100 : 25 + task_id_new * 100]
                    indice2=data_[50+task_id_new * 100 : 75 + task_id_new * 100]
                if index ==1:
                    indice1=data_[task_id_new * 100 : 25 + task_id_new * 100]
                    indice2=data_[75+task_id_new * 100 : 100 + task_id_new * 100]
                if index ==2:
                    indice1=data_[task_id_new * 100 : 25 + task_id_new * 100]
                    indice2=data_[75+task_id_new * 100 : 100 + task_id_new * 100]
                if index ==3:
                    indice1=data_[20+task_id_new * 100 : 50 + task_id_new * 100]
                    indice2=data_[60+task_id_new * 100 : 80 + task_id_new * 100]
                data=np.concatenate([indice1, indice2], axis=0)
            datas.append(data)
            labels.append(np.full((len(data)), label))
            self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
            if len(self.last) == args.num_clients :
                self.last=[]
                self.task_id = task_id_new
        return labels

    def getSampleData(self, classes, exemplar_set, exemplar_label_set, group):
        datas, labels = [], []
        if len(exemplar_set) != 0 and len(exemplar_label_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            length = len(datas[0])
            labels = [np.full((length), label) for label in exemplar_label_set]

        if group == 0:
            for label in classes:
                data = self.data[np.array(self.targets) == label]
                datas.append(data)
                labels.append(np.full((data.shape[0]), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)

    def getTrainItem(self, index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return index, img, target

    def getTestItem(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]

        if self.test_transform:
            img = self.test_transform(img)

        if self.target_test_transform:
            target = self.target_test_transform(target)

        return index, img, target

    def __getitem__(self, index):
        if self.TrainData != []:
            return self.getTrainItem(index)
        elif self.TestData != []:
            return self.getTestItem(index)

    def __len__(self):
        if self.TrainData != []:
            return len(self.TrainData)
        elif self.TestData != []:
            return len(self.TestData)

        # 获得标签label的数据集

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]


