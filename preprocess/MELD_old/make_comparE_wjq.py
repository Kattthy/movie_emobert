import os
import h5py
import json
import numpy as np
import pandas as pd
import scipy.signal as spsig
from tqdm import tqdm

CONFIG = {
    "audio_dir": "/data8/hzp/datasets/MELD_process/audio_clips",
    "data_dir": "/data8/hzp/datasets/MELD.Raw",
    "target_dir": "/data8/hzp/datasets/MELD_target",
    "output_dir": "/data8/hzp/datasets/MELD_feature"
}

class ComParEExtractor(object):
    ''' 抽取comparE特征, 输入音频路径, 输出npy数组, 每帧130d
    '''
    def __init__(self, opensmile_tool_dir=None, downsample=10, tmp_dir='.tmp', no_tmp=False):
        ''' Extract ComparE feature
            tmp_dir: where to save opensmile csv file
            no_tmp: if true, delete tmp file
        '''
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        if opensmile_tool_dir is None:
            opensmile_tool_dir = '/data8/hzp/emo_bert/tools/opensmile-2.3.0'
        self.opensmile_tool_dir = opensmile_tool_dir
        self.tmp_dir = tmp_dir
        self.downsample = downsample
        self.no_tmp = no_tmp
    
    def __call__(self, wav):
        basename = os.path.basename(wav).split('.')[0]
        save_path = os.path.join(self.tmp_dir, basename+".csv")
        cmd = 'SMILExtract -C {}/config/ComParE_2016.conf \
            -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1 \
            -I {} -lldcsvoutput {} -instname xx -O ? -noconsoleoutput 1'
        os.system(cmd.format(self.opensmile_tool_dir, wav, save_path))
        try:
            df = pd.read_csv(save_path, delimiter=';')
            wav_data = df.iloc[:, 2:]
        except:
            return np.zeros([1,130])
        if len(wav_data) == 0:
            return np.zeros([1,130])
        #print(basename, wav_data)
        if self.downsample > 0:
            if len(wav_data) > self.downsample:
                wav_data = spsig.resample_poly(wav_data, up=1, down=self.downsample, axis=0)
                if self.no_tmp:
                    os.remove(save_path) 
            else:               
                wav_data = np.array([[0]*130])
                print(wav_data.shape)
                print(f'Error in {wav}, no feature extracted')

        return wav_data


def get_trn_val_tst(target_root_dir, setname):
    int2name = np.load(os.path.join(target_root_dir, '{}_int2name.npy'.format(setname)))
    int2label = np.load(os.path.join(target_root_dir, '{}_label.npy'.format(setname)))
    print(int2name)
    assert len(int2name) == len(int2label)
    return int2name, int2label

def make_all_comparE(config, downsample): # 是否需要降采样
    extractor = ComParEExtractor(downsample=downsample)
    trn_int2name, trn_int2label = get_trn_val_tst(config['target_dir'], 'trn')
    val_int2name, val_int2label = get_trn_val_tst(config['target_dir'], 'val')
    tst_int2name, tst_int2label = get_trn_val_tst(config['target_dir'], 'tst')
    trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
    val_int2name = list(map(lambda x: x[0].decode(), val_int2name))
    tst_int2name = list(map(lambda x: x[0].decode(), tst_int2name))
    trn_int2label = list(trn_int2label)
    val_int2label = list(val_int2label)
    tst_int2label = list(tst_int2label)
    #all_utt_ids = trn_int2name + val_int2name + tst_int2name
    if downsample > 0:
        save_dir = os.path.join(config['output_dir'], 'A', 'comparE_downsample')
    else:
        save_dir = os.path.join(config['output_dir'], 'A', 'comparE')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    train_h5f = h5py.File(os.path.join(save_dir, 'trn.h5'), 'w')
    
    for idx,utt_id in enumerate(trn_int2name):
        wav_path = os.path.join(config['audio_dir'],'train',utt_id+'.wav')
        feat = extractor(wav_path)
        '''
        if feat is None:
            trn_int2name.remove(utt_id)
            trn_int2label.pop(idx)
            print(utt_id,'has been removed')
        else:
        '''
        train_h5f[utt_id] = feat
    '''
    all_name_list = np.array([np.array([utt_id.encode()]) for utt_id in trn_int2name])
    all_label_list = np.array([label for label in trn_int2label])

    np.save(os.path.join(config['target_dir'], 'trn_int2name1.npy'), all_name_list)
    np.save(os.path.join(config['target_dir'], 'trn_label1.npy'), all_label_list)
    '''
    
    

    
    test_h5f = h5py.File(os.path.join(save_dir, 'tst.h5'), 'w')
    for idx,utt_id in enumerate(tst_int2name):
        wav_path = os.path.join(config['audio_dir'],'test',utt_id+'.wav')
        feat = extractor(wav_path)
        '''
        if feat is None:
            tst_int2name.remove(utt_id)
            tst_int2label.pop(idx)
            print(utt_id,'has been removed')
        else:
        '''
        test_h5f[utt_id] = feat
    '''
    all_name_list = np.array([np.array([utt_id.encode()]) for utt_id in tst_int2name])
    all_label_list = np.array([label for label in tst_int2label])

    np.save(os.path.join(config['target_dir'], 'tst_int2name.npy'), all_name_list)
    np.save(os.path.join(config['target_dir'], 'tst_label.npy'), all_label_list)
    '''
    
    val_h5f = h5py.File(os.path.join(save_dir, 'val.h5'), 'w')
    for idx,utt_id in enumerate(val_int2name):
        wav_path = os.path.join(config['audio_dir'],'dev',utt_id+'.wav')
        feat = extractor(wav_path)
        '''
        if feat is None:
            val_int2name.remove(utt_id)
            val_int2label.pop(idx)
            print(utt_id,'has been removed')
        else:
        '''
        val_h5f[utt_id] = feat
    #all_name_list = np.array([np.array([utt_id.encode()]) for utt_id in val_int2name])
    #all_label_list = np.array([label for label in val_int2label])

    #np.save(os.path.join(config['target_dir'], 'val_int2name.npy'), all_name_list)
    #np.save(os.path.join(config['target_dir'], 'val_label.npy'), all_label_list)
    


def normlize_on_trn(config, input_file, output_file): #在每个特征元素位置，对所有训练集数据中的所有帧的该位置元素取平均以及计算标准差
    h5f = h5py.File(output_file, 'w')
    in_data = h5py.File(input_file, 'r')

    trn_int2name, _ = get_trn_val_tst(config['target_dir'], 'trn')
    trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
    print(in_data['dia230_utt0'][()])
    all_feat = [in_data[utt_id][()] for utt_id in trn_int2name] #3个维度：[音频数据数, seq_len, ft_dim]
    all_feat = np.concatenate(all_feat, axis=0) #这里是将不同个音频的数据拼起来，拼完之后第0维代表包括了所有音频数据的每一帧。也就是说这时有2个维度：[所有音频合起来的seq_len, ft_dim]
    mean_f = np.mean(all_feat, axis=0)
    std_f = np.std(all_feat, axis=0)
    std_f[std_f == 0.0] = 1.0
    group = h5f.create_group('trn') #创建一个名为'trn'的组
    group['mean'] = mean_f
    group['std'] = std_f
    print("mean:", np.sum(mean_f))
    print("std:", np.sum(std_f))

def statis_comparE(config, downsample): #特征的统计量（最小长度、最大长度以及几分位长度）
    if downsample > 0:
        path = os.path.join(config['feature_root'], 'A', 'comparE_downsample', 'trn.h5')
    else:
        path = os.path.join(config['feature_root'], 'A', 'comparE', 'trn.h5')
    h5f = h5py.File(path, 'r')
    lengths = []
    for utt_id in h5f.keys():
        lengths.append(h5f[utt_id][()].shape[0])
    lengths = sorted(lengths)
    print('MIN:', min(lengths))
    print('MAX:', max(lengths))
    print('MEAN: {:.2f}'.format(sum(lengths) / len(lengths)))
    print('50%:', lengths[len(lengths)//2])
    print('75%:', lengths[int(len(lengths)*0.75)])
    print('90%:', lengths[int(len(lengths)*0.9)])
    
    
if __name__ == '__main__':
    #pwd = os.path.abspath(__file__)
    #pwd = os.path.dirname(pwd)
    #config_path = os.path.join('/data1/wjq/MMIN-emo-Audio_comparE/data/config/IEMOCAP_config.json')
    #config = json.load(open(config_path))
    downsample = 10
    if downsample > 0:
        comparE_name = 'comparE_downsample'
    else:
        comparE_name = 'comparE'
    make_all_comparE(CONFIG,downsample)
    normlize_on_trn(CONFIG, os.path.join(CONFIG['output_dir'], 'A', comparE_name, 'trn.h5'), os.path.join(CONFIG['output_dir'], 'A', comparE_name, 'mean_std.h5'))