from torchvision.datasets import MNIST
import numpy as np
import torch
from PIL import Image
import random
import torch.nn.init as init
from collections import Counter


class MNIST10(MNIST):
    def __init__(self,root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=True):
        super(MNIST10,self).__init__(root,
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
      
    def concatenate(self, datas, labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1, len(datas)):
            con_data=np.concatenate((con_data, datas[i]), axis=0)
            con_label=np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, classes):
        datas,labels=[],[]
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TestData, self.TestLabels=self.concatenate(datas, labels)

    def getTrainData(self, index,task_id_new, classes, exemplar_set, exemplar_label_set):
        # 获得存储器里面的数据和标签
        datas, labels = [], []
        if len(exemplar_set) != 0 and len(exemplar_label_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            length = len(datas[0])  # 即m,每个exemplar中包含有m个同一类别的样本
            labels = [np.full((length), label) for label in exemplar_label_set]

        for label in classes:
            data_ = self.data[np.array(self.targets) == label]
           # data =  random.sample(list(data_), 500)
            data = data_[index * 300+task_id_new*1200 :index * 300 + 300 + task_id_new*1200]
            datas.append(data)  # [1,1,1],
            labels.append(np.full((len(data)), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
        return labels

    def getSampleData(self, classes, exemplar_set, exemplar_label_set, group):
        datas,labels=[],[]
        if len(exemplar_set)!=0 and len(exemplar_label_set)!=0:
            datas=[exemplar for exemplar in exemplar_set]
            length=len(datas[0])
            labels=[np.full((length), label) for label in exemplar_label_set]

        if group == 0:
            for label in classes:
                data=self.data[np.array(self.targets)==label]
                datas.append(data)
                labels.append(np.full((data.shape[0]),label))
        self.TrainData, self.TrainLabels=self.concatenate(datas,labels)

    def getTrainItem(self,index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]

        if self.transform:
            img=self.transform(img)

        if self.target_transform:
            target=self.target_transform(target)

        return index,img,target

    def getTestItem(self,index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]

        if self.test_transform:
            img=self.test_transform(img)

        if self.target_test_transform:
            target=self.target_test_transform(target)

        return index, img, target

    def __getitem__(self, index):
        if self.TrainData!=[]:
            return self.getTrainItem(index)
        elif self.TestData!=[]:
            return self.getTestItem(index)


    def __len__(self):
        if self.TrainData!=[]:
            return len(self.TrainData)
        elif self.TestData!=[]:
            return len(self.TestData)

    # 获得标签label的数据集
    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label]


