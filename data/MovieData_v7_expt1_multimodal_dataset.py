import os
import json
from typing import List
import torch
import numpy as np
import h5py
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from data.base_dataset import BaseDataset

from random import shuffle


class MovieDataV7Expt1MultimodalDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--A_type', type=str, help='which audio feat to use')
        parser.add_argument('--V_type', type=str, help='which visual feat to use')
        parser.add_argument('--L_type', type=str, help='which lexical feat to use')
        parser.add_argument('--output_dim', type=int, help='how many label types in this dataset')
        parser.add_argument('--norm_method', type=str, choices=['utt', 'trn'], help='how to normalize input comparE feature')
        return parser
    
    def __init__(self, opt, set_name):
        ''' MovieData_v7 dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)

        # record & load basic settings 
        cvNo = opt.cvNo
        self.set_name = set_name
        pwd = os.path.abspath(__file__)
        pwd = os.path.dirname(pwd)
        config = json.load(open(os.path.join(pwd, 'config', 'MovieData_v7_expt_config.json')))
        self.norm_method = opt.norm_method
        # load feature
        self.A_type = opt.A_type
        self.all_A = h5py.File(os.path.join(config['feature_root'], 'A', f'{self.A_type}', 'all.h5'), 'r')
        if self.A_type == 'comparE_downsample':
            self.mean_std = h5py.File(os.path.join(config['feature_root'], 'A', 'comparE_downsample', 'mean_std.h5'), 'r')
            self.mean = torch.from_numpy(self.mean_std[str(cvNo)]['mean'][()]).unsqueeze(0).float()
            self.std = torch.from_numpy(self.mean_std[str(cvNo)]['std'][()]).unsqueeze(0).float()
        elif self.A_type == 'comparE':
            self.mean_std = h5py.File(os.path.join(config['feature_root'], 'A', 'comparE', 'mean_std.h5'), 'r')
            self.mean = torch.from_numpy(self.mean_std[str(cvNo)]['mean'][()]).unsqueeze(0).float()
            self.std = torch.from_numpy(self.mean_std[str(cvNo)]['std'][()]).unsqueeze(0).float()
        self.V_type = opt.V_type
        self.all_V = h5py.File(os.path.join(config['feature_root'], 'V', f'{self.V_type}', 'all.h5'), 'r')
        self.L_type = opt.L_type
        self.all_L = h5py.File(os.path.join(config['feature_root'], 'L', f'{self.L_type}', 'all.h5'), 'r')
        # load target
        label_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_label.npy")
        int2name_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_int2name.npy")
        self.label = np.load(label_path)
        self.label = np.argmax(self.label, axis=1)
        self.int2name = np.load(int2name_path)
        
        #---------筛选训练集数据----------
        if set_name == 'trn': #是训练集，需要筛选neutral和happiness类别数据

            emo2id_dict = {'anger': 0, 'happiness': 1, 'neutral': 2, 'sadness': 3}
            id2emo_dict = {0: 'anger', 1: 'happiness', 2: 'neutral', 3: 'sadness'}

            assert(len(self.label) == len(self.int2name))

            all_dict = {}
            for i in range(4):
                all_dict[id2emo_dict[i]] = []

            for i, j in zip(self.int2name, self.label):
                all_dict[id2emo_dict[j]].append({'name': i, 'label': j})

            train_data = [] #尽量控制类别均衡的训练集数据
            for i in range(4):
                shuffle(all_dict[id2emo_dict[i]]) #打乱各类别中的数据顺序
                if id2emo_dict[i] == 'neutral':
                    train_data.extend(all_dict[id2emo_dict[i]][:3000])
                elif id2emo_dict[i] == 'happiness':
                    train_data.extend(all_dict[id2emo_dict[i]][:3000])
                else:
                    train_data.extend(all_dict[id2emo_dict[i]])

            shuffle(train_data)

            self.label = np.array([i['label'] for i in train_data])
            self.int2name = np.array([i['name'] for i in train_data])
        
        #--------------------------------

        self.manual_collate_fn = True
        print(f"MovieData_v7 dataset {set_name} created with total length: {len(self)}")

    def __getitem__(self, index):
        int2name = self.int2name[index][0].decode()
        label = torch.tensor(self.label[index])
        # process A_feat
        A_feat = torch.from_numpy(self.all_A[int2name][()]).float()
        if self.A_type == 'comparE':
            A_feat = self.normalize_on_utt(A_feat) if self.norm_method == 'utt' else self.normalize_on_trn(A_feat)
        # process V_feat 
        V_feat = torch.from_numpy(self.all_V[int2name][()]).float()
        # proveee L_feat
        L_feat = torch.from_numpy(self.all_L[int2name][()]).float()
        return {
            'A_feat': A_feat, 
            'V_feat': V_feat,
            'L_feat': L_feat,
            'label': label,
            'int2name': int2name
        }
    
    def __len__(self):
        return len(self.label)
    
    def normalize_on_utt(self, features):
        mean_f = torch.mean(features, dim=0).unsqueeze(0).float()
        std_f = torch.std(features, dim=0).unsqueeze(0).float()
        std_f[std_f == 0.0] = 1.0
        features = (features - mean_f) / std_f
        return features
    
    def normalize_on_trn(self, features):
        features = (features - self.mean) / self.std
        return features

    def collate_fn(self, batch):
        A = [sample['A_feat'] for sample in batch]
        V = [sample['V_feat'] for sample in batch]
        L = [sample['L_feat'] for sample in batch]
        lengths = torch.tensor([len(sample) for sample in A]).long()
        A = pad_sequence(A, batch_first=True, padding_value=0)
        V = pad_sequence(V, batch_first=True, padding_value=0)
        L = pad_sequence(L, batch_first=True, padding_value=0)
        label = torch.tensor([sample['label'] for sample in batch])
        int2name = [sample['int2name'] for sample in batch]
        return {
            'A_feat': A, 
            'V_feat': V,
            'L_feat': L,
            'label': label,
            'lengths': lengths,
            'int2name': int2name
        }

if __name__ == '__main__':
    class test:
        cvNo = 1
        A_type = "comparE"
        V_type = "denseface"
        L_type = "bert_large"
        norm_method = 'trn'

    
    opt = test()
    print('Reading from dataset:')
    a = MovieDataV7Expt1MultimodalDataset(opt, set_name='trn')
    data = next(iter(a))
    for k, v in data.items():
        if k not in ['int2name', 'label']:
            print(k, v.shape)
        else:
            print(k, v)
    print('Reading from dataloader:')
    x = [a[100], a[34], a[890]]
    print('each one:')
    for i, _x in enumerate(x):
        print(i, ':')
        for k, v in _x.items():
            if k not in ['int2name', 'label']:
                print(k, v.shape)
            else:
                print(k, v)
    print('packed output')
    x = a.collate_fn(x)
    for k, v in x.items():
        if k not in ['int2name', 'label']:
            print(k, v.shape)
        else:
            print(k, v)
    