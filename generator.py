import torch
import torch.nn as nn
import torch.nn.functional as F
MAXLOG = 0.1
from torch.autograd import Variable
import collections
import numpy as np
from model_config import GENERATORCONFIGS
from option import args_parser
import math
import torch.nn.init as init

args = args_parser()
class Generator(nn.Module):
    def __init__(self,dataset=args.dataset, model='cnn', embedding=False, latent_layer_idx=-1):
        super(Generator, self).__init__()
        self.embedding = embedding
        self.dataset = dataset
        self.n_teacher_iters = 5
        #self.model=model
        self.latent_layer_idx = latent_layer_idx
        self.hidden_dim, self.latent_dim, self.input_channel, self.n_class, self.noise_dim = GENERATORCONFIGS[dataset]
        self.input_dim = self.noise_dim * 2 if self.embedding else self.noise_dim + self.n_class
        #self.input_dim =self.n_class
        self.fc_configs = [self.input_dim, self.hidden_dim]
        self.init_loss_fn()
        self.build_network()

    def get_number_of_parameters(self):
        pytorch_total_params=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params


    def init_loss_fn(self):
        self.crossentropy_loss=nn.NLLLoss(reduce=False) # same as above
        self.diversity_loss = DiversityLoss(metric='l1')
        self.dist_loss = nn.MSELoss()
    def build_network(self):
        if self.embedding:
            self.embedding_layer = nn.Embedding(self.n_class, self.noise_dim)
        ### FC modules ####
        self.fc_layers = nn.ModuleList()
        act = nn.Sigmoid
        for i in range(len(self.fc_configs) - 1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i + 1]   #fc_configs = [input_dim, self.hidden_dim]
            #self.input_dim, out_dim = self.fc_configs[i], self.fc_configs[i+1]
            print("Build layer {} X {}".format(self.input_dim, out_dim))
            #self.fc = nn.Linear(input_dim, out_dim, bias=True)
            self.fc2 = nn.Linear(self.input_dim, out_dim, bias=True).cuda()
            #self.bn = nn.BatchNorm1d(out_dim,eps = 1e-6,track_running_stats=False).cuda()
            self.act =  nn.LeakyReLU(0.2, inplace=True).cuda()
            #self.representation_layer = nn.Linear(out_dim, self.latent_dim).cuda()
            #self.fc_layers += [self.fc.cuda(), self.bn.cuda(), self.act.cuda(),self.representation_layer.cuda()]#一层cnn+bn+relu
            #self.fc_layers += [self.fc.cuda(), self.act.cuda(), self.representation_layer.cuda()]  # 一层cnn+bn+relu
            self.fc_layers += [self.fc2.cuda(), self.act]
            ### Representation layer
        #self.representation_layer = nn.Linear(self.fc_configs[-1], self.latent_dim).cuda()#hidden_dim->latent_dim
        self.representation_layer = nn.Linear(out_dim, self.latent_dim).cuda()  # hidden_dim->latent_dim
        print("Build last layer {} X {}".format(out_dim, self.latent_dim))
    def Incremental_learning(self, numclass):
        # 全连接层的权重，偏置，输入特征（不变），输出特征（原类别数，需改变）
        weight = self.fc2.weight.data
        bias = self.fc2.bias.data
        in_feature = self.fc2.in_features
        out_feature = self.fc2.out_features  # 原类别数

        self.n_class = numclass
        #self.embedding_layer = nn.Embedding(self.n_class, self.noise_dim)
        self.input_dim = self.noise_dim + self.n_class
        #self.input_dim = self.n_class
        idx = 0
        for i, layer in enumerate(self.fc_layers):
            if layer is self.fc2:  # 判断对象是否相同
                idx = i
                break
        new_fc = nn.Linear(in_features=self.input_dim, out_features=out_feature)  # 定义新的 nn.Linear 层对象
        new_fc.weight.data[:, :in_feature] = weight
        #new_fc.weight.data[:out_feature] = weight
        new_fc.bias.data[:] = bias
        self.fc_layers[idx] = new_fc.cuda()  # 用新的对象替换旧的对象，并将其移动到 GPU 上
    def forward(self, labels, latent_layer_idx=-1, verbose=True):
        """
        G(Z|y) or G(X|y):
        Generate either latent representation( latent_layer_idx < 0) or raw image (latent_layer_idx=0) conditional on labels.
        根据标签生成潜像表示(潜像层索引< 0)或原始图像(潜像层索引=0)。
        :param labels:
        :param latent_layer_idx:
            if -1, generate latent representation of the last layer,
            -2 for the 2nd to last layer, 0 for raw images.
        :param verbose: also return the sampled Gaussian noise if verbose = True
        :return: a dictionary of output information.
        """
        result = {}
        batch_size = labels.shape[0]
        eps = torch.rand((batch_size, self.noise_dim)) # sampling from Gaussian  得到噪声矩阵eps
        if verbose:
            result['eps'] = eps
        if self.embedding: # embedded dense vector    两种不同编码方式
            y_input = self.embedding_layer(labels)#embedding_layer = nn.Embedding(self.n_class, self.noise_dim)
        else: # one-hot (sparse) vector
            y_input = torch.FloatTensor(batch_size, self.n_class)
            y_input.zero_()
            # labels = labels.view
            labels = labels.type(torch.int64)
            y_input.scatter_(1, labels.view(-1, 1).cpu(), 1).cuda()
        z = torch.cat((eps, y_input), dim=1)  # 加噪声
        #z = y_input.cuda()

        ### FC layers
        for layer in self.fc_layers:
            z = layer(z.cuda())#fc_layers += [fc, bn, act] 全连接、batchnorm、激活函数

        #z = self.generator(z.cuda()).cuda()
        z = self.representation_layer(z) #representation_layer = nn.Linear(self.fc_configs[-1], self.latent_dim) 隐藏层维度映射到潜在层维度
        result['output'] = z
        return result  # result['eps'] = eps ，result['output'] = z   z：潜在层的向量



    @staticmethod
    def normalize_images(layer):
        """
        Normalize images into zero-mean and unit-variance.
        """
        mean = layer.mean(dim=(2, 3), keepdim=True)
        std = layer.view((layer.size(0), layer.size(1), -1)) \
            .std(dim=2, keepdim=True).unsqueeze(3)
        return (layer - mean) / std

def train_generator(available_label,selected_users,model_all, generative_optimizer ,generative_lr_scheduler,gen,model_g,batch_size=128, epoches=10, latent_layer_idx=-1, verbose=False):
    """
    Learn a generator that find a consensus latent representation z, given a label 'y'.
    :param batch_size:
    :param epoches:
    :param latent_layer_idx: if set to -1 (-2), get latent representation of the last (or 2nd to last) layer.
    :param verbose: print loss information.
    :return: Do not return anything.
    """
    ensemble_alpha = 1
    ensemble_eta = 0
    ensemble_beta = 0
    generative_model = gen
    qualified_labels = available_label
    TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS = 0, 0, 0
    selected_users=selected_users
    def update_generator_(n_iters, selected_users,generative_model , generative_optimizer, generative_lr_scheduler,model_all, student_model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS):
        generative_model.train()
        for i in range(n_iters):
            generative_optimizer.zero_grad()
            diversity_loss = 0
            y = np.random.choice(qualified_labels, batch_size)  # 随机选择batchsize个标签y
            y_input = torch.LongTensor(y)  # 变成张量形式
            ## feed to generator
            gen_result = generative_model(y_input, latent_layer_idx=latent_layer_idx, verbose=True)
            # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
            gen_output, eps = gen_result['output'], gen_result['eps']
            ##### get losses ####
            #diversity_loss = generative_model.diversity_loss(eps, gen_output)  # encourage different outputs
            ######### get teacher loss ############
            teacher_loss = 0
            teacher_logit = 0
            #for user_idx in (selected_users):

            for j in range(len(selected_users)):
                gen_result = generative_model(y_input, latent_layer_idx=latent_layer_idx, verbose=True)
                gen_output, eps = gen_result['output'], gen_result['eps']
                user_result_given_gen = model_all[j](gen_output, index=-1)
                user_output_logp_ = F.log_softmax(user_result_given_gen, dim=1)
                teacher_loss_ = torch.mean( \
                    generative_model.crossentropy_loss(user_output_logp_.cuda(), y_input.cuda()))
                teacher_loss += teacher_loss_
                teacher_logit += user_result_given_gen


            #correct, total = 0, 0
            ######### get student loss ############
            #student_output = student_model(gen_output, index=-1)
            #student_loss = F.kl_div(F.log_softmax(student_output, dim=1),F.softmax(teacher_logit, dim=1))
            #correct += (user_output_logp_.cpu() == y_input.cpu()).sum()
            #total += len(labels)
            #accuracy = 100 * correct / total
            #print('client:', j, '    accuracy:', accuracy)
            if ensemble_beta > 0:
                loss = ensemble_alpha * teacher_loss - ensemble_beta * student_loss + ensemble_eta * diversity_loss
            else:
                loss = ensemble_alpha * teacher_loss + ensemble_eta * diversity_loss
            #with amp.scale_loss(loss,generative_optimizer) as scaled_loss:
            #   scaled_loss.backward()
            loss.backward()
            nn.utils.clip_grad_norm_(generative_model.parameters(), max_norm=0.1)
            generative_optimizer.step()
            generative_lr_scheduler.step()
            TEACHER_LOSS += ensemble_alpha * teacher_loss  # (torch.mean(TEACHER_LOSS.double())).item()
            #STUDENT_LOSS += ensemble_beta * student_loss  # (torch.mean(student_loss.double())).item()
            #DIVERSITY_LOSS += ensemble_eta * diversity_loss  # (torch.mean(diversity_loss.double())).item()
        return TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS
    for i in range(epoches):
        TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS = update_generator_(
            20,selected_users,generative_model ,generative_optimizer,generative_lr_scheduler,model_all, model_g,TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
    TEACHER_LOSS = TEACHER_LOSS.cpu().detach().numpy() / (epoches*len(selected_users))
    #STUDENT_LOSS = STUDENT_LOSS.cpu().detach().numpy() / (epoches)
    #DIVERSITY_LOSS = DIVERSITY_LOSS.cpu().detach().numpy() / (epoches)
    #info = "Generator: Teacher Loss= {:.4f}, Student Loss= {:.4f}, Diversity Loss = {:.4f}, ". \
    #    format(TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
    info = "Generator: Teacher Loss= {:.4f} ".format(TEACHER_LOSS)
    #if verbose:
    if args.local_rank == 0:
        print(info)


class DivLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self):
        """
        Class initializer.
        """
        super().__init__()

    def forward2(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        chunk_size = layer.size(0) // 2

        ####### diversity loss ########
        eps1, eps2=torch.split(noises, chunk_size, dim=0)
        chunk1, chunk2=torch.split(layer, chunk_size, dim=0)
        lz=torch.mean(torch.abs(chunk1 - chunk2)) / torch.mean(
            torch.abs(eps1 - eps2))
        eps=1 * 1e-5
        diversity_loss=1 / (lz + eps)
        return diversity_loss

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer=layer.view((layer.size(0), -1))
        chunk_size=layer.size(0) // 2

        ####### diversity loss ########
        eps1, eps2=torch.split(noises, chunk_size, dim=0)
        chunk1, chunk2=torch.split(layer, chunk_size, dim=0)
        lz=torch.mean(torch.abs(chunk1 - chunk2)) / torch.mean(
            torch.abs(eps1 - eps2))
        eps=1 * 1e-5
        diversity_loss=1 / (lz + eps)
        return diversity_loss


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """
    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist.cuda() * layer_dist.cuda()))
