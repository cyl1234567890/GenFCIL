from torchvision.datasets import SVHN
import numpy as np
import torch
from PIL import Image
import random
import torch.nn.init as init
import scipy.io
import cv2

class ISVHN(torch.utils.data.Dataset):
    def __init__(self, root,  train_transform=None, test_transform=None, download=False, split='train'):
        super(ISVHN, self).__init__()
        #self.dataset = SVHN(root=root, transform=transform, target_transform=target_transform, download=download, split=split)
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []
        self.train_data = None
        self.train_targets = None
        self.test_data = None
        self.test_targets = None

    def get_data(self):


        # 加载SVHN数据集并进行处理，根据需要筛选特定类别的样本等
        train_path = '/feng_data/cyl/FCIL---now/src/svhn/train_32x32.mat'
        test_path = '/feng_data/cyl/FCIL---now/src/svhn/test_32x32.mat'
        train_list_img, train_list_label, test_list_img, test_list_label = [], [], [], []
        train_data = scipy.io.loadmat(train_path)
        train_images = train_data['X']
        train_labels = train_data['y']

        # Load testing data
        test_data = scipy.io.loadmat(test_path)
        test_images = test_data['X']
        test_labels = test_data['y']

        # Add training images and labels to lists
        for i in range(train_images.shape[3]):
            img = train_images[:, :, :,  i]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = train_labels[i][0] - 1  # Subtract 1 to match the class index (0-based)

            train_list_img.append(img)
            train_list_label.append(label)
        # Add testing images and labels to lists
        for i in range(test_images.shape[3]):
            img = test_images[:, :, :, i]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = test_labels[i][0] - 1  # Subtract 1 to match the class index (0-based)
            test_list_img.append(img)
            test_list_label.append(label)

        train_list_img, test_list_img = np.asarray(train_list_img), np.asarray(test_list_img)
        train_list_label, test_list_label = np.asarray(train_list_label), np.asarray(test_list_label)

        self.train_data, self.test_data = train_list_img, test_list_img
        self.train_targets, self.test_targets = train_list_label, test_list_label
    def getTestData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.test_data[np.array(self.test_targets) == label]
            #data =data_[:1000]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))

        self.TestData, self.TestLabels = self.concatenate(datas, labels)
        self.TrainData, self.TrainLabels = [], []

    def getTrainData(self, index, task_id_new, classes, exemplar_set, exemplar_label_set):
        datas, labels = [], []
        if len(exemplar_set) != 0 and len(exemplar_label_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            length = len(datas[0])  # 即m,每个exemplar中包含有m个同一类别的样本
            labels = [np.full((length), label), label in exemplar_label_set]

        for label in classes:
            data_ = self.train_data[np.array(self.train_targets) == label]
            data = data_[index * 200 + task_id_new * 1000:index * 200 + 300 + task_id_new * 1000]
            datas.append(data)
            labels.append(np.full((len(data)), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
        return labels

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTrainItem(self, index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]

        if self.train_transform:
            img = self.train_transform(img)
        return index, img, target

    def getTestItem(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]
        if self.test_transform:
            img = self.test_transform(img)
        return index, img, target

    def __getitem__(self, index):
        if self.TrainData != []:
            return self.getTrainItem(index)
        elif self.TestData != []:
            return self.getTestItem(index)

    def __len__(self):
        if self.TrainData != []:
            return len(self.TrainData)
        elif self.TestData !=[]:
            return len(self.TestData)

    def get_image_class(self, label):
        return self.train_data[np.array(self.train_targets) == label]
