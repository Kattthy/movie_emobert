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

def get_trn_val_tst(target_root_dir, setname):
    int2name = np.load(osp.join(target_root_dir, '{}_int2name.npy'.format(setname)))
    int2label = np.load(osp.join(target_root_dir, '{}_label.npy'.format(setname)))
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
    face_root = os.path.join(config['data_root'], 'faces')
    target_root = config['target_root']

    trn_int2name, _ = get_trn_val_tst(target_root, 'trn')
    val_int2name, _ = get_trn_val_tst(target_root, 'val')
    tst_int2name, _ = get_trn_val_tst(target_root, 'tst')
    #all_utt_ids = trn_int2name + val_int2name + tst_int2name
    for utt_id in tqdm(trn_int2name):
        face_dir = os.path.join(config['data_root'], 'faces', 'train', utt_id, utt_id+'_aligned')
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
    config_path = '/data8/hzp/movie_emobert/data/config/MELD_config.json'
    config = json.load(open(config_path))
    calc_mean_std(config)