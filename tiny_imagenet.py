import numpy as np
from PIL import Image
import cv2
import os
import pandas as pd
from option import args_parser

args = args_parser()

class Tiny_Imagenet:
    def __init__(self, root, train_transform=None, test_transform=None):
        super(Tiny_Imagenet, self).__init__()
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
        self.root = root

    def get_data(self):
        train_path = '.svhn/train_32x32.mat'
        test_path = '.svhn/test_32x32.mat'

        # Load training data
        train_data = scipy.io.loadmat(train_path)
        train_images = train_data['X']
        train_labels = train_data['y']

        # Load testing data
        test_data = scipy.io.loadmat(test_path)
        test_images = test_data['X']
        test_labels = test_data['y']

        class_counts = {}  # Initialize a dictionary to keep track of the number of samples per class

        # Add training images and labels to lists
        for i in range(train_images.shape[3]):
            img = train_images[:, :, :, i].transpose((2, 0, 1))
            label = train_labels[i][0] - 1  # Subtract 1 to match the class index (0-based)

            train_list_img.append(img)
            train_list_label.append(label)

        # Add testing images and labels to lists
        for i in range(test_images.shape[3]):
            img = test_images[:, :, :, i].transpose((2, 0, 1))
            label = test_labels[i][0] - 1  # Subtract 1 to match the class index (0-based)

            if label not in class_counts:
                class_counts[label] = 0

            if class_counts[label] < 1000:  # Only add samples if the class count is less than 1000
                test_list_img.append(img)
                test_list_label.append(label)
                class_counts[label] += 1


    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.test_data[np.array(self.test_targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))

        self.TestData, self.TestLabels = self.concatenate(datas, labels)
        self.TrainData, self.TrainLabels = [], []

    def getTrainData(self, classes, exemplar_set, exemplar_label_set):
        datas, labels = [], []
        if len(exemplar_set) != 0 and len(exemplar_label_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            length = len(datas[0])
            labels = [np.full((length), label) for label in exemplar_label_set]

        for label in classes:
            data = self.train_data[np.array(self.train_targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
        return labels

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
        elif self.TestData != []:
            return len(self.TestData)

    def get_image_class(self, label):
        return self.train_data[np.array(self.train_targets) == label]


