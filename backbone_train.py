from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import visdom
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms, datasets, models
import torch.nn.functional as F

from resnet import resnet32
import cifar10 as dataset
gpu_status = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# Optimization options
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
# Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Method options
parser.add_argument('--percent', type=float, default=0,
                    help='Percentage of noise')
parser.add_argument('--train_ratio', type=float, default=0.9,
                    help='Percentage of train')

parser.add_argument('--alpha', type=float, default=1.0,
                    help='Hyper parameter alpha of loss function')
parser.add_argument('--beta', type=float, default=0.5,
                    help='Hyper parameter beta of loss function')
parser.add_argument('--lamda', type=float, default=1000,
                    help='Hyper parameter beta of loss function')
parser.add_argument('--asym', action='store_true',
                    help='Asymmetric noise')
parser.add_argument('--out', default='./save_model',
                    help='Directory to output the result')

args = parser.parse_args()
class_num=10
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

viz=visdom.Visdom()
line = viz.line(Y=np.arange(args.epochs))
line2 = viz.line(Y=np.arange(args.epochs))
def main():
    # start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    best_acc = 0.0
    if not os.path.exists(args.out):
        os.mkdir(args.out)

    # Data
    print(f'==> Preparing {"asymmetric" if args.asym else "symmetric"} nosiy cifar10')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),


    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset, valset = dataset.get_cifar10('./data', args, train=True, download=False, transform_train=transform_train,
                                           transform_val=transform_val)
    data_sizes=int(len(trainset))
    print('trainset,valset',len(trainset),len(valset))
    val_dataSizes=int(len(valset))
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)##################
    valloader = data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    print("==> creating preact_resnet")
    model = resnet32()
    # if os.path.exists('./save_model/resnet32.pth'):
    #     model.load_state_dict(torch.load('./save_model/resnet32.pth'))
    #     print('load resnet32.pth successfully')
    # else:
    #     print('load resnet32.pth failed')

    if gpu_status:
        model = model.cuda()

    cudnn.benchmark = True #可加快速度
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    start_time=time.time()
    train_loss, test_loss_v, train_acc, test_acc_v, time_p ,corr_rate_all,epoch_num= [], [], [], [], [],[],[]
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0.0
        # labels_grad = np.zeros((data_sizes, 10), dtype=np.float32)

        for batch_idx, (inputs, labels, indexs, labels_update, gtrue_labels) in enumerate(trainloader):
            if gpu_status:
                inputs, labels = inputs.cuda(), labels.cuda()
                gtrue_labels=gtrue_labels.cuda()
           
            # compute output
            outputs = model(inputs)
            loss=criterion(outputs,labels)
            preds = torch.max(outputs.detach(), 1)[1]#不需要梯度，得到下标
            optimizer.zero_grad()
            loss.backward()
          
            optimizer.step()#更新参数
            running_loss += loss.item()*len(labels)#标量 用item()得到python数字
            running_corrects += torch.sum(preds == gtrue_labels.detach())
        scheduler.step()
        epoch_loss = running_loss / data_sizes

        if gpu_status:
            epoch_acc = running_corrects.cpu().numpy() / data_sizes
        else:
            epoch_acc = running_corrects.numpy() / data_sizes

        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        time_elapsed = time.time() - start_time
        time_p.append(time_elapsed)
        print("[{}/{} epoches] train_loss:{:.4f}||train_acc:{:.4f}||time passed:{:.0f}m {:.0f}s".format(epoch + 1,
                                                                                                        args.epochs,
                                                                                                        train_loss[-1],
                                                                                                        train_acc[-1],
                                                                                                        time_elapsed // 60,
                                                                                                        time_elapsed % 60))

        # validate
        test_loss, test_acc = validate(model, valloader, criterion, epoch, val_dataSizes)

        test_loss_v.append(test_loss)
        test_acc_v.append(test_acc)
        epoch_num.append(epoch + 1)
        viz.line(X=np.column_stack((np.array(epoch_num), np.array(epoch_num))),
                 Y=np.column_stack((np.array(train_loss), np.array(test_loss_v))),
                 win=line,
                 opts=dict(xlabel='epoch',ylabel='Loss',legend=["train_loss", "val_loss"],
                           title="30% symmetric noise: ResNet-32 VAL Loss:{:.4f}".format(test_loss_v[-1])))

        viz.line(X=np.column_stack((np.array(epoch_num), np.array(epoch_num))),
                 Y=np.column_stack((np.array(train_acc), np.array(test_acc_v))),
                 win=line2,
                 opts=dict(xlabel='epoch',ylabel='accuracy',legend=[ "train_acc", "val_acc"],
                           title="30% symmetric noise: ResNet-32 VAL ACC:{:.4f}".format(test_acc_v[-1])))

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),os.path.join(args.out,'sym_3_resnet32.pth'))
            print('resnet32 saved')


def validate(model,valloader,criterion, epoch,data_sizes):
    start_time=time.time()
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(valloader):

            if gpu_status:
                inputs, labels = inputs.cuda(), labels.cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.max(outputs.data, 1)[1]  ####

            running_loss += loss.item() * len(labels)  # 标量
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / data_sizes
        if gpu_status:
            epoch_acc = running_corrects.cpu().numpy() / data_sizes
        else:
            epoch_acc = running_corrects.numpy() / data_sizes


        time_elapsed = time.time() - start_time
        print("[{}/{} epoches] val_loss:{:.4f}||val_acc:{:.4f}||time cost:{:.0f}m {:.0f}s".format(epoch + 1,args.epochs,
                                                                                                        epoch_loss,epoch_acc,
                                                                                                        time_elapsed // 60,time_elapsed % 60))
        return epoch_loss,epoch_acc

def pencil_loss(outputs,labels_update,labels):
    # sfm=nn.Softmax(1)
    # pred=sfm(outputs)
    # pred = pred.detach()#detach使得pred requires_grad=False,并且不影响outputs

    pred = F.softmax(outputs, dim=1)####是否应该detach????????????/


    # criterion = nn.CrossEntropyLoss()
    # Lo=criterion(labels_update,labels)

    Lo = -torch.mean(F.log_softmax(labels_update, dim=1)[torch.arange(labels_update.shape[0]),labels])

    # Le=criterion(outputs,pred)
    Le = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * pred, dim=1))

    # Lc=criterion(labels_update,pred)-criterion(outputs,pred)
    Lc = -torch.mean(torch.sum(F.log_softmax(labels_update, dim=1) * pred, dim=1)) - Le
    loss_total = Lc /class_num+args.alpha* Lo +args.beta* Le /class_num
    return loss_total

if __name__ == '__main__':
    main()
