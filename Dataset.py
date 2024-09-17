import os
import re
import h5py

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import random_split

class MD_detection(Dataset):
    def __init__(self,root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = ['T','H']
        self.data = np.empty((0,128,45))
        self.label = np.empty((0,1))
 
        self.label_count = {key:0 for key in self.classes}
        
        self.filtered_files = []

        for folder in root:
            for file in os.listdir(os.path.join(folder,'md_h5_files')):
                gt = re.split(r'_',file)[1][0] #H,T
                if gt in ['H','T']:
                    if self._checkMovingTarget(file):
                        self.filtered_files.append(file)
                        with h5py.File(os.path.join(folder,'md_h5_files',file),'r') as f:
                            tensor = f["tensor"][:]
                            if gt[0] == 'H':
                                tensor = tensor[1::2,:,:]
                            self.data = np.concatenate((self.data,tensor),axis=0)
                            label = np.zeros((tensor.shape[0],1))
                            label = np.add(label, self.classes.index(gt))

                            self.label = np.concatenate((self.label,label),axis=0)
                            self.label_count[gt] = self.label_count[gt] + tensor.shape[0]
    
    def _checkMovingTarget(self,file):
        if len(re.split(r'_',file)) > 3:
            if re.split(r'_',file)[3][0] in list('abcde'):
                return True
            else:
                return False
        else:
            return False

    def __len__(self):
        return self.label.shape[0]
    
    def __getitem__(self,index):
        data = self.data[index,:,:]
        label = self.label[index]
        if self.transform:
            data = self.transform(data)
        # return {'data': data,
        #         'label':torch.tensor(label)}
        return {'data': data,
                'label': self._get_sample_label(int(label))}
    
    def _get_sample_label(self,label):
        one_hot_label = F.one_hot(torch.tensor(label), num_classes=len(self.classes))
        return one_hot_label

    

if __name__ == "__main__":
    # root = ['Sep14']
    # root = [os.path.join(r'G:\Jiarui',f) for f in root]

    #test_dataset = MD_detection(root,classes)
    env_list = ['Sep14','Sep15']
    env_list = [os.path.join(r'G:\Jiarui',f) for f in env_list]
                             

    dataset = MD_detection(env_list)
    print(dataset.label_count)
    train, val= random_split(dataset, [int(0.7*len(dataset)), len(dataset)-int(0.7*len(dataset))])
    print(len(train), len(val))
    print(train[0]['data'].shape, train[0]['label'].shape)
    print(val[0]['label'])
    print('---------------------------------')
    num_positive = torch.tensor([0,0]) 
    for i in range(len(val)):
        num_positive = num_positive + val[i]['label']
    print(num_positive)
