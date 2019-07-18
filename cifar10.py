import numpy as np
import random
from PIL import Image
import torchvision
import argparse
import torch.nn.functional as F
import torch
import visdom
viz=visdom.Visdom()

K=10

def get_cifar10(root, args, train=True,
                transform_train=None, transform_val=None,
                download=False):
    base_dataset = torchvision.datasets.CIFAR10(root, train=train, download=download)
    # print('base_dataset',dir(base_dataset))

    train_idxs, val_idxs = train_val_split(base_dataset.targets,args.train_ratio)##注意，若为0.4版本，应为base_dataset.targets

    train_dataset = CIFAR10_train(root, train_idxs, args, train=train, transform=transform_train)
    if args.asym:
        train_dataset.asymmetric_noise()
        # print('origin labels_update',np.argmax(train_dataset.labels_update[0:128],axis=1))# 7 7 3 7

    else:
        train_dataset.symmetric_noise()
    val_dataset = CIFAR10_val(root, val_idxs, train=train, transform=transform_val)

    print(f"Train: {len(train_idxs)} Val: {len(val_idxs)}")
    return train_dataset, val_dataset

def train_val_split(train_val,train_ratio):
    train_val = np.array(train_val)
    train_n = int(len(train_val) * train_ratio / 10)#5000
    train_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(train_val == i)[0]
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:train_n])
        val_idxs.extend(idxs[train_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs


class CIFAR10_train(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, args=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_train, self).__init__(root, train=train,
                                            transform=transform, target_transform=target_transform,
                                            download=download)
        self.args = args
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.true_labels= self.targets+1-1
            # print('true_labels',self.true_labels[0:128])
            # print('len(self.data)',len(self.data))
            self.labels_update = np.zeros((len(self.data), 10), dtype=np.float32)
            # self.soft_labels[np.arange(len(self.data)),self.targets]=1
            # self.labels_update = self.soft_labels * 10

        self.lamda=args.lamda
        self.prediction = np.zeros((len(self.data), 10, 10), dtype=np.float32)
        self.count = 0
        self.count_img=0
        self.best_corr_rate=0.0
        self.ch_label=np.zeros(10,dtype=np.float32)#每个类有多少张图片标签变成噪声标签

    def symmetric_noise(self):
        indices = np.random.permutation(len(self.data))#不改变原数组
        # print('len(self.data)',len(self.data))
        # print('indices',indices.shape)
        for i, idx in enumerate(indices):
            if i < self.args.percent * len(self.data):
                self.ch_label[self.targets[idx]]+=1
                self.targets[idx] = np.random.randint(10, dtype=np.int32)
            self.labels_update[idx][self.targets[idx]] = K

    def asymmetric_noise(self):
        for i in range(10):
            indices = np.where(self.true_labels == i)[0]
            # print('indices',indices,indices.shape)
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.args.percent * len(indices):
                    # truck -> automobile
                    if i == 9:
                        self.targets[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.targets[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.targets[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.targets[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.targets[idx] = 7
                self.labels_update[idx][self.targets[idx]] = K
        # print('origin asymmetric_noise',np.mean(np.argmax(self.labels_update[0:128],axis=1)==self.targets[0:128]))

    def label_update(self, lamda,labels_grad):
        self.count += 1

        self.labels_update = self.labels_update - self.lamda * labels_grad
        if self.lamda>50:
            self.lamda = self.lamda - 50
        corr_labels = F.softmax(torch.from_numpy(self.labels_update), dim=1)
        corr_labels=corr_labels.numpy()

        corr_rate = sum(np.argmax(corr_labels, axis=1) == self.true_labels)/len(self.true_labels)
        if corr_rate > self.best_corr_rate:
            self.best_corr_rate = corr_rate
            print('self.best_corr_rate',self.best_corr_rate)
            np.save(f'{self.args.out}/images_sym7.npy', self.data)
            np.save(f'{self.args.out}/corr_labels_sym7.npy', corr_labels)
            np.save(f'{self.args.out}/noise_labels_sym7.npy', self.targets)
            np.save(f'{self.args.out}/true_labels_sym7.npy', self.true_labels)
            np.save(f'{self.args.out}/labels_update_sym7.npy', self.labels_update)
        return corr_rate

    def reload_label(self):
        self.data = np.load(f'{self.args.out}/images_sym7.npy')
        self.targets = np.load(f'{self.args.out}/corr_labels_sym7.npy')
        # self.true_labels = np.load(f'{self.args.out}/true_labels.npy')
        # self.labels_update = np.load(f'{self.args.out}/labels_update.npy')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        self.count_img+=1
        img, target = self.data[index], self.targets[index]#img:32,32,3
        labels_update = self.labels_update[index]
        true_labels = self.true_labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # if self.count_img == 1:
        #     print('img.shape',img.shape)
        #     img_show=np.transpose(img,[2,0,1])
        #     viz.image(img_show,env='show_img')
        #     viz.image(img_show+random.gauss(0,0.1),env='show_img')

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # print('target',target)
        if self.target_transform is not None:
            target = self.target_transform(target)
            # labels_update = self.target_transform(labels_update)
        # print('getitem',target==np.argmax(labels_update))
        # print('target after',target)
        return img, target, index,labels_update,true_labels


class CIFAR10_val(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_val, self).__init__(root, train=train,
                                          transform=transform, target_transform=target_transform,
                                          download=download)

        self.data = self.data[indexs]
        self.targets = np.array(self.targets)[indexs]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
    # Optimization options
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
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
    parser.add_argument('--begin', type=int, default=70,
                        help='When to begin updating labels')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Hyper parameter alpha of loss function')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Hyper parameter beta of loss function')
    parser.add_argument('--asym', action='store_true',
                        help='Asymmetric noise')
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')

    args = parser.parse_args()
    get_cifar10('./data',args)