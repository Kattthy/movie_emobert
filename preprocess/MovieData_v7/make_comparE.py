import os
import h5py
import json
import numpy as np
import pandas as pd
import scipy.signal as spsig
from tqdm import tqdm


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
            #opensmile_tool_dir = '/root/opensmile-2.3.0/'
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
        
        df = pd.read_csv(save_path, delimiter=';')
        wav_data = df.iloc[:, 2:]
        if self.downsample > 0:
            if len(wav_data) > self.downsample:
                wav_data = spsig.resample_poly(wav_data, up=1, down=self.downsample, axis=0)
                if self.no_tmp:
                    os.remove(save_path) 
            else:
                wav_data = None
                print(f'Error in {wav}, no feature extracted')

        return wav_data

def get_trn_val(target_root_dir, cv, setname):
    int2name = np.load(os.path.join(target_root_dir, str(cv), '{}_int2name.npy'.format(setname)))
    int2label = np.load(os.path.join(target_root_dir, str(cv), '{}_label.npy'.format(setname)))
    assert len(int2name) == len(int2label)
    return int2name, int2label

def make_all_comparE(config, downsample):
    extractor = ComParEExtractor(downsample=downsample)
    trn_int2name, _ = get_trn_val(config['target_root'], 1, 'trn')
    val_int2name, _ = get_trn_val(config['target_root'], 1, 'val')
    trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
    val_int2name = list(map(lambda x: x[0].decode(), val_int2name))
    all_utt_ids = trn_int2name + val_int2name
    if downsample > 0:
        save_dir = os.path.join(config['feature_root'], 'A', 'comparE_downsample')
    else:
        save_dir = os.path.join(config['feature_root'], 'A', 'comparE')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    all_h5f = h5py.File(os.path.join(save_dir, 'all.h5'), 'w')
    for utt_id in tqdm(all_utt_ids):
        movie_id = utt_id.split('_')[0]
        clip_id = utt_id.split('_')[1].split('.')[0]
        wav_path = os.path.join(config['data_root'], 'audio', movie_id, movie_id + '_' + clip_id + '.wav')
        feat = extractor(wav_path)
        all_h5f[utt_id] = feat

def normlize_on_trn(config, input_file, output_file): #在每个特征元素位置，对所有训练集数据中的所有帧的该位置元素取平均以及计算标准差
    h5f = h5py.File(output_file, 'w')
    in_data = h5py.File(input_file, 'r')
    for cv in range(1, 2):
        trn_int2name, _ = get_trn_val(config['target_root'], cv, 'trn')
        trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
        all_feat = [in_data[utt_id][()] for utt_id in trn_int2name] #3个维度：[音频数据数, seq_len, ft_dim]
        all_feat = np.concatenate(all_feat, axis=0) #这里是将不同个音频的数据拼起来，拼完之后第0维代表包括了所有音频数据的每一帧。也就是说这时有2个维度：[所有音频合起来的seq_len, ft_dim]
        mean_f = np.mean(all_feat, axis=0)
        std_f = np.std(all_feat, axis=0)
        std_f[std_f == 0.0] = 1.0
        cv_group = h5f.create_group(str(cv))
        cv_group['mean'] = mean_f
        cv_group['std'] = std_f
        print(cv)
        print("mean:", np.sum(mean_f))
        print("std:", np.sum(std_f))


def statis_comparE(config, downsample): #特征的统计量（最小长度、最大长度以及几分位长度）
    if downsample > 0:
        path = os.path.join(config['feature_root'], 'A', 'comparE_downsample', 'all.h5')
    else:
        path = os.path.join(config['feature_root'], 'A', 'comparE', 'all.h5')
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
    downsample = -1 #上面的代码中downsample>0的情况下才会进行downsample，所以这里是没有用它的。
    #downsample = 10
    if downsample > 0:
        comparE_name = 'comparE_downsample'
    else:
        comparE_name = 'comparE'
    pwd = os.path.abspath(__file__) #获取当前文件的绝对路径
    pwd = os.path.dirname(pwd) #获取该文件的上级目录
    pwd = os.path.dirname(pwd) #获取该文件的上级目录
    config_path = os.path.join(pwd, '../', 'data/config', 'MovieData_v7_config.json')
    config = json.load(open(config_path))
    make_all_comparE(config, downsample)
    normlize_on_trn(config, os.path.join(config['feature_root'], 'A', comparE_name, 'all.h5'), os.path.join(config['feature_root'], 'A', comparE_name, 'mean_std.h5'))