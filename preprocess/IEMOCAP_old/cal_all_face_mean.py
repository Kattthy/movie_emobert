import os
import os.path as osp
import shutil
import h5py
import numpy as np
import cv2
from tqdm import tqdm
import math

def get_trn_val_tst(target_root_dir, cv, setname):
    int2name = np.load(osp.join(target_root_dir, str(cv), '{}_int2name.npy'.format(setname)))
    int2label = np.load(osp.join(target_root_dir, str(cv), '{}_label.npy'.format(setname)))
    int2name = [x[0].decode() for x in int2name]
    assert len(int2name) == len(int2label)
    return int2name, int2label

def calc_mean_std():
    img_size = 64

    all_faces = []
    face_root = '/data1/hzp/IEMOCAP_frames_and_faces/'
    target_root = '/data1/hzp/IEMOCAP_features_npy/target'

    
    trn_int2name, _ = get_trn_val_tst(target_root, 1, 'trn')
    val_int2name, _ = get_trn_val_tst(target_root, 1, 'val')
    tst_int2name, _ = get_trn_val_tst(target_root, 1, 'tst')
    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    

    """
    trn_int2name, _ = get_trn_val_tst(target_root, 10, 'trn')
    all_utt_ids = trn_int2name
    """


    for utt_id in tqdm(all_utt_ids):
        ses_id = utt_id[4]
        utt_dir = osp.join(face_root, 'Session' + ses_id, 'face', utt_id)
        utt_faces = [osp.join(utt_dir, i) for i in os.listdir(utt_dir)]
        for face in utt_faces:
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




calc_mean_std()

"""
pixel_num_list = [755232768, 755232768, 786497536, 786497536, 767217664, 767217664, 773296128, 773296128, 748482560, 748482560]
mean_list = [124.75667333465064, 124.75667333465064, 129.76274068581444, 129.76274068581444, 133.44022035446775, 133.44022035446775, 126.14515766461979, 126.14515766461979, 134.80633526718378, 134.80633526718378]
std_list = [47.2135120118084, 47.2135120118084, 48.812542236498565, 48.812542236498565, 47.72318928826184, 47.72318928826184, 46.12535548927472, 46.12535548927472, 48.80030617372965, 48.80030617372965]

N_pixel = 0
mean = 0
for i, j in zip(pixel_num_list, mean_list):
    N_pixel += i
    mean += i*j
mean /= N_pixel

print(N_pixel) #7661453312
print(mean) #算出来是129.76750764020056
"""




#所有数据（训练+测试+验证）：
#像素数：957681664
#mean：129.76750764020056
#std：47.904685096526066

#------------------------
#训练集1：
#像素数：755232768
#mean：124.75667333465064
#std：47.2135120118084

#训练集2：
#像素数：755232768
#mean：124.75667333465064
#std：47.2135120118084

#训练集3：
#像素数：786497536
#mean：129.76274068581444
#std：48.812542236498565

#训练集4：
#像素数：786497536
#mean：129.76274068581444
#std：48.812542236498565

#训练集5：
#像素数：767217664
#mean：133.44022035446775
#std：47.72318928826184

#训练集6：
#像素数：767217664
#mean：133.44022035446775
#std：47.72318928826184

#训练集7：
#像素数：773296128
#mean：126.14515766461979
#std：46.12535548927472

#训练集8：
#像素数：773296128
#mean：126.14515766461979
#std：46.12535548927472

#训练集9：
#像素数：748482560
#mean：134.80633526718378
#std：48.80030617372965

#训练集10：
#像素数：748482560
#mean：134.80633526718378
#std：48.80030617372965
