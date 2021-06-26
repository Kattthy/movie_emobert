import os
import os.path as osp
import shutil
import h5py
import numpy as np
import cv2
from tqdm import tqdm
import math
import json
import glob

def get_trn_val(target_root_dir, cvNo, setname):
    int2name = np.load(osp.join(target_root_dir, str(cvNo), '{}_int2name.npy'.format(setname)))
    int2label = np.load(osp.join(target_root_dir, str(cvNo), '{}_label.npy'.format(setname)))
    int2name = [x[0].decode() for x in int2name]
    assert len(int2name) == len(int2label)
    return int2name, int2label

def get_faces(face_dir):
    face_lst = glob.glob(os.path.join(face_dir, '*.bmp')) #face_dir为某个clip的人脸图像目录
    face_lst = sorted(face_lst)
    return face_lst

def calc_mean_std(config):
    img_size = 64

    all_faces = []
    face_root = os.path.join(config['data_root'], 'face')
    target_root = config['target_root']

    trn_int2name, _ = get_trn_val(target_root, 1, 'trn')
    val_int2name, _ = get_trn_val(target_root, 1, 'val')
    all_utt_ids = trn_int2name + val_int2name

    for utt_id in tqdm(all_utt_ids):
        movie_id = utt_id.split('_')[0]
        clip_id = utt_id.split('_')[1].split('.')[0]
        face_dir = os.path.join(face_root, movie_id, movie_id + '_' + clip_id)
        face_lst = get_faces(face_dir)
        for face in face_lst:
            img = cv2.imread(face)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (img_size, img_size))
            img = img.reshape(-1)
            all_faces.append(img)

    all_faces = np.concatenate(all_faces)
    print(all_faces.shape)

    mean = all_faces.mean()
    std = all_faces.std()

    print('MEAN:', mean)
    print('STD:', std)

    #记录下运算的结果：
    record_file = os.path.join(config['data_root'], 'face_mean_std.txt')
    with open(record_file, 'w') as f:
        f.write('mean:' + str(mean) + '\n')
        f.write('std:' + str(std) + '\n')




if __name__ == '__main__':
    pwd = os.path.abspath(__file__) #获取当前文件的绝对路径
    pwd = os.path.dirname(pwd) #获取该文件的上级目录
    pwd = os.path.dirname(pwd) #获取该文件的上级目录
    config_path = os.path.join(pwd, '../', 'data/config', 'MovieData_v7_complete_config.json')
    config = json.load(open(config_path))
    calc_mean_std(config)