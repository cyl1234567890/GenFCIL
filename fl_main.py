from GLFC import GLFC_model
from ResNet import *
import torch
from torch import nn
import copy
import random
import os.path as osp
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
from myNetwork import network, LeNet
from Fed_utils import * 
from ProxyServer import *
from mini_imagenet import *
from tiny_imagenet import *
from MNIST10 import *
from ICIFAR10 import *
from ISVHN import *
from option import args_parser
import torch.distributed as dist
import torchvision.models as models
from generator import Generator,train_generator
from Fed import *
from func import *



#import DataParallel
from torch.utils.data import DataLoader, DistributedSampler

def main():
    args = args_parser()

    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

    feature_extractor = resnet18_cbam()
    num_clients = args.num_clients
    old_client_0 = []
    models = []
    available_labels = []
    oldmodel=None

    setup_seed(args.seed)
    gen = Generator()
    model_g = network(args.numclass, feature_extractor)
    model_g = model_to_device(model_g, False, device, device_ids)


    generative_optimizer = torch.optim.Adam(params=gen.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08,weight_decay=0, amsgrad=False)
    generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=generative_optimizer, gamma=0.98)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    mnist_train_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    mnist_test_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])

    if args.dataset == 'cifar100':
        train_dataset = iCIFAR100('./dataset', transform=train_transform, download=False)
        test_dataset = iCIFAR100('./dataset', test_transform=test_transform, train=False, download=False)
    elif args.dataset == 'svhn':
        train_dataset = ISVHN('./svhn',  train_transform=train_transform, download=False, split='train')
        train_dataset.get_data()
        #test_dataset = train_dataset
        test_dataset = ISVHN('./svhn', test_transform=test_transform, download=False, split='test')
        test_dataset.get_data()

    elif args.dataset == 'mnist':
        train_dataset = MNIST10('./dataset_mnist', transform=mnist_train_transform, download=True)
        test_dataset = MNIST10('./dataset_mnist', test_transform=mnist_test_transform, train=False, download=True)
    elif args.dataset == 'cifar10':
        train_dataset = ICIFAR10('./dataset_cifar10', transform=train_transform, download=False)
        test_dataset = ICIFAR10('./dataset_cifar10', test_transform=test_transform, train=False, download=False)
    #else:
    #    train_dataset = Mini_Imagenet('./train', train_transform=train_transform, test_transform=test_transform)
    #    train_dataset.get_data()
    #    test_dataset = train_dataset

    for i in range(args.num_clients):
        model_temp = GLFC_model(i, args.numclass, feature_extractor, gen, args.batch_size, args.task_size, args.memory_size,
                                args.epochs_local, args.learning_rate, train_dataset, device, oldmodel, device_ids)
        model_temp.model = model_to_device(model_temp.model, False, device, device_ids)
        models.append(model_temp)


    ## training log
    output_dir = osp.join('./training_log', str(args.dataset))
    if not osp.exists(output_dir):
        os.system('mkdir -p ' + output_dir)  # 命令行的方式创建目录
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    out_file = open(osp.join(output_dir, 'log_tar_' + str(args.task_size) + 'learning_rate' + str(args.learning_rate) + 'batch_size'+ str(args.batch_size)+ 'clusting:' +str(args.clusting) + '.txt'), 'w')
    log_str = 'method_{}, task_size_{}, learning_rate_{}, iid_level_{}'.format(args.method, args.task_size, args.learning_rate, args.iid_level)
    out_file.write(log_str + '\n')
    out_file.flush()

    #classes_learned = args.task_size
    old_task_id = -1
    model_old = [None, None]
    z = tensor = torch.zeros(10)
    for ep_g in range(args.epochs_global):
        #if args.dataset !='cifar100':
        available_labels=[]
        task_id = ep_g // args.tasks_global
        #classes_learned = args.numclass + args.increase_class * task_id
        '''
        if task_id != old_task_id and old_task_id != -1:
            overall_client = len(old_client_0) + len(old_client_1) + len(new_client)
            new_client = [i for i in range(overall_client, overall_client + 5)]
            old_client_1 = random.sample([i for i in range(overall_client)], int(overall_client * 0.8))
            old_client_0 = [i for i in range(overall_client) if i not in old_client_1]
            num_clients = len(new_client) + len(old_client_1) + len(old_client_0)
            print((overall_client,new_client,old_client_1,old_task_id),num_clients,task_id)
        '''
        #if task_id != old_task_id and old_task_id != -1:
        #    classes_learned += args.task_size
            #model_g.Incremental_learning(classes_learned)
            #gen.Incremental_learning(classes_learned)
        #    model_g = model_g.to(device)
            #oldmodel = copy.deepcopy(model_g)
        for model in models:
            model.model = copy.deepcopy(model_g)

        if args.local_rank == 0:
            print('communication：{}, task_id: {}'.format(ep_g, task_id))
        w_local = []
        model_all = []
        linear=[]
        label_set=[]
        feature_local=[]
        clients_index=select_users(num_clients, args.local_clients)

        if args.local_rank == 0:
            print('selsct client:', clients_index)
        for c in clients_index:
            local_model,local_linear, clients_model,feature, available_label= local_train(models, c,  model_g,task_id,oldmodel,model_old, ep_g, old_client_0, ep_g,z)
            print(available_label)
            available_labels.extend(available_label)
            label_set.append(available_label)
            w_local.append(local_model)
            linear.append(local_linear)
            feature_local.extend(feature)
            model_all.append(clients_model)

        available_labels=list(set(available_labels))
        train_generator(available_labels, clients_index, model_all, generative_optimizer, generative_lr_scheduler, gen,model_g)
        w_g_new = Fed_Avg(w_local)
        if (ep_g %10 ==0 and ep_g!=0) or (ep_g==9):
        #if (ep_g % 10 == 0 ):
            z = Fed_DeltaPoolCls(linear, label_set, feature_local)
            #print(z)
        w_g_new = Fed_Avg(w_local)
        model_g.load_state_dict(w_g_new)
        participant_exemplar_storing(models, num_clients, model_g, old_client_0, task_id, clients_index)
        acc_global = model_global_eval(z,model_g, test_dataset, args.numclass, ep_g, args.task_size, device, device_ids,0)
        print('Task: {}, Round: {} Accuracy = {:.2f}% '.format(task_id, ep_g, acc_global))

        '''
        acc_ = model_select(model_g, test_dataset, task_id, args.task_size, device, device_ids)
        best_acc = acc_
        num=-1
        j=-1
        for i,c in enumerate(clients_index):
            acc_ = model_select(model_all[i], test_dataset, task_id, args.task_size, device, device_ids)
            if acc_> best_acc:
                best_acc = acc_
                num = c
                j=i

        if num == -1:
            model_last = copy.deepcopy(model_g)
        else:
            model_last = copy.deepcopy(model_all[j])
        if args.local_rank == 0:
            print('Task: {}, Round: {} ,client:{} Accuracy = {:.2f}% '.format(task_id, ep_g, num,best_acc))
        '''
        log_str = 'Task: {}, Round: {} Accuracy = {:.2f}%'.format(task_id, ep_g, acc_global)
        out_file.write(log_str + '\n')
        out_file.flush()
        if old_task_id != task_id and task_id > 0:
            model_old[0] = copy.deepcopy(model_old[1])

        # 更新任务id
        old_task_id = task_id
        if ep_g % 10 == 0:
            model_old[1] = copy.deepcopy(model_g)
        #if ep_g % 10 == 0:
        #    oldmodel=copy.deepcopy(model_g)
        #if ep_g % 10 == 9:
        #    oldmodel =None
        #else:

        oldmodel =copy.deepcopy(model_g)




if __name__ == '__main__':
    main()
