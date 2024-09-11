import torch.nn as nn
import torch

# 将feature_extractor连接numclass
class network(nn.Module):

    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor

        self.fc = nn.Linear(feature_extractor.fc.in_features, numclass,bias= True)


    def forward(self, input,index=0):
        if index != -1:
            #mask = torch.ones(input.size())
            input = self.feature(input) # 得到（B, C)
        input = input.cuda()
        self.fc = self.fc.cuda()
        input = self.fc(input.cuda())  # 得到（B, numclass)
        return input

    def Incremental_learning(self, numclass):
        # 全连接层的权重，偏置，输入特征（不变），输出特征（原类别数，需改变）
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features  # 原类别数

        # 将最后一层全连接层重置为新的输出类别
        self.fc = nn.Linear(in_feature, numclass, bias=True)
        # 原先神经元的权重偏置不变
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

    def feature_extractor(self, inputs):
        return self.feature(inputs)

    def predict(self, fea_input):
        return self.fc(fea_input)


class LeNet(nn.Module):

    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def weights_init(m):
    try:
        if hasattr(m, "weight"):  # 检查m对象中是否有weight属性，初始化
            m.weight.data.uniform_(-0.5, 0.5)  # 从均匀分布（-0.5， 0.5）中抽样得到的值填充weight,下同
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())
