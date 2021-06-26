import os
import glob
import torch
import h5py
import cv2
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/data8/hzp/movie_emobert/preprocess')
from tools.denseface_tf.dense_net import DenseNet
import tensorflow as tf
import collections
import json

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

def get_trn_val(target_root_dir, cv, setname):
    int2name = np.load(os.path.join(target_root_dir, str(cv), '{}_int2name.npy'.format(setname)))
    int2label = np.load(os.path.join(target_root_dir, str(cv), '{}_label.npy'.format(setname)))
    assert len(int2name) == len(int2label)
    return int2name, int2label

class DensefaceExtractor(object):
    ''' 抽取denseface特征, mean是数据集图片的均值, std是数据集图片的方差(灰度图)
        device表示使用第几块gpu(从0开始计数)
    '''
    #def __init__(self, mean=131.0754, std=47.858177, device=0, smooth=False, logger=None): #待会重新算一下mean和std
    def __init__(self, restore_path=None, mean=63.5211, std=41.3223, device=1):
        """ extract densenet feature
            Parameters:
            ------------------------
            model: model class returned by function 'load_model'
        """
        if restore_path is None:
            restore_path = '/data8/hzp/emo_bert/tools/denseface/pretrained_model/model/epoch-200'
        self.model = self.load_model(restore_path)
        self.mean = mean
        self.std = std
        self.dim = 342                  # returned feature dim
        self.device = device
    
    def load_model(self, restore_path):
        # fake data_provider
        growth_rate = 12
        img_size = 64
        depth = 100
        total_blocks = 3
        reduction = 0.5
        keep_prob = 1.0
        bc_mode = True
        model_path = restore_path
        dataset = 'FER+'
        num_class = 8

        DataProvider = collections.namedtuple('DataProvider', ['data_shape', 'n_classes'])
        data_provider = DataProvider(data_shape=(img_size, img_size, 1), n_classes=num_class)
        model = DenseNet(data_provider=data_provider, growth_rate=growth_rate, depth=depth,
                        total_blocks=total_blocks, keep_prob=keep_prob, reduction=reduction,
                        bc_mode=bc_mode, dataset=dataset)

        end_points = model.end_points
        model.saver.restore(model.sess, model_path)
        print("Successfully load model from model path: {}".format(model_path))
        return model

    def get_faces(self, face_dir):
        face_lst = glob.glob(os.path.join(face_dir, '*.bmp')) #face_dir为某个clip的人脸图像目录
        face_lst = sorted(face_lst)
        return face_lst

    def get_face_tensors(self, face_lst):
        img_size = 64
        images_mean = self.mean #训练集所有图像的均值
        images_std = self.std #训练集所有图像的标准差
        imgs = []
        for face in face_lst:
            img = cv2.imread(face)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (img_size, img_size))
            imgs.append(img)
        imgs = (np.array(imgs, np.float32) - images_mean) / images_std #对人脸图像进行归一化
        imgs = np.expand_dims(imgs, 3) #扩充一个维度作为channel维
        return imgs

    def chunk(self, lst, chunk_size): #chunk的作用：如果一个目录下的人脸图片过多，超过32张，就每32张截断一次
        idx = 0
        while chunk_size * idx < len(lst):
            yield lst[idx*chunk_size: (idx+1)*chunk_size]
            idx += 1

    def record_mean_std(self, config): #将本次抽取特征时指定的训练集均值和方差记录到h5pf文件当中
        '''
        目前这里没分cv，因为我默认在划分验证集的情况下，只计算一次训练集+验证集+测试集的均值和方差，而不是对每个划分的训练集分别计算一次
        如果是对每个划分的训练集分别计算一次，那么这里也要分别存每个划分的不同统计量。
        '''
        mean_std_file = os.path.join(config['feature_root'], 'V', 'denseface', 'mean_std.h5')
        h5f = h5py.File(mean_std_file, 'w')
        group = h5f.create_group('trn') #创建一个名为'trn'的组
        group['mean'] = self.mean
        group['std'] = self.std
        print("mean:", self.mean)
        print("std:", self.std)


    def __call__(self, config):
        trn_int2name, _ = get_trn_val(config['target_root'], 1, 'trn')
        val_int2name, _ = get_trn_val(config['target_root'], 1, 'val')
        trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
        val_int2name = list(map(lambda x: x[0].decode(), val_int2name))
        all_utt_ids = trn_int2name + val_int2name
        mkdir(os.path.join(config['feature_root'], 'V', 'denseface'))
        all_h5f = h5py.File(os.path.join(config['feature_root'], 'V', 'denseface', 'all.h5'), 'w')
        self.record_mean_std(config) #将本次抽取特征时指定的训练集均值和方差记录到h5pf文件当中
        for utt_id in tqdm(all_utt_ids):
            movie_id = utt_id.split('_')[0]
            clip_id = utt_id.split('_')[1].split('.')[0]
            face_dir = os.path.join(config['data_root'], 'face', movie_id, movie_id + '_' + clip_id)
            face_lst = self.get_faces(face_dir)
            if len(face_lst) == 0:
                all_h5f[utt_id] = np.zeros((1, 342))
                continue
            face_tensor = self.get_face_tensors(face_lst) #face_tensor = torch.from_numpy(self.get_face_tensors(face_lst)).to(device)
            #这一步当中已经对读入的face进行了归一化

            utt_feat = []
            for face_tensor_bs in self.chunk(face_tensor, 32): #chunk的作用：如果一个目录下的人脸图片过多，超过32张，就每32张截断一次。截断的目的应该是在于控制batch_size的大小
                with tf.device('/gpu:{}'.format(self.device)):
                    feed_dict = {
                        self.model.images: face_tensor_bs,
                        self.model.is_training: False
                    }
                    feat = self.model.sess.run(self.model.end_points['fc'], feed_dict=feed_dict) #feat.shape：[截断完之后的片段faces数量，342]
                    utt_feat.append(feat)
            utt_feat = np.concatenate(utt_feat, axis=0) #这一步相当于又把截断的那个维度接回来了。utt_feat.shape：[原片段faces数量，342]
            all_h5f[utt_id] = feat

"""
def padding_to_fixlen(data, length):
    if len(data) >= length:
        ret = data[:length]
    else:
        ret = np.concatenate([data, np.zeros([length-len(data), data.shape[1]])], axis=0)
    return ret

def migrate_denseface_to_npy(config):
    max_len = 22
    all_ft = h5py.File(os.path.join(config['feature_root'], 'V', 'denseface', 'all.h5'), "r")
    
    save_dir = os.path.join(config['feature_root'], 'V', 'denseface')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for set_name in ['trn', 'val']:
        int2name, _ = get_trn_val(config['target_root'], set_name)
        int2name = list(map(lambda x: x[0].decode(), int2name))
        fts = []
        for utt_id in int2name:
            ft = all_ft[utt_id][()]
            ft = padding_to_fixlen(ft, max_len)
            fts.append(ft)
        fts = np.array(fts)
        print(f'{set_name} {fts.shape}')
        np.save(os.path.join(save_dir, f'{set_name}.npy'), fts)
"""




if __name__ == '__main__':
    pwd = os.path.abspath(__file__) #获取当前文件的绝对路径
    pwd = os.path.dirname(pwd) #获取该文件的上级目录
    pwd = os.path.dirname(pwd) #获取该文件的上级目录
    config_path = os.path.join(pwd, '../', 'data/config', 'MovieData_v7_complete_config.json')
    config = json.load(open(config_path))
    run_denseface = DensefaceExtractor() #加载denseface模型
    run_denseface(config)