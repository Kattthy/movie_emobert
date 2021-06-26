import os, glob
import os.path as osp
import subprocess
import numpy as np
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm
from functools import partial

CONFIG = {
    "audio_dir": "/data7/lrc/movie_dataset/output/MELD_audio/",
    "data_dir": "/data2/ljj/MELD.Raw",
    "target_dir": "/data6/lrc/MELD/target",
    "output_dir": "/data6/lrc/MELD/feature"
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
            opensmile_tool_dir = '/root/opensmile-2.3.0/'
        self.opensmile_tool_dir = opensmile_tool_dir
        self.tmp_dir = tmp_dir
        self.downsample = downsample
        self.no_tmp = no_tmp
    
    def __call__(self, wav):
        # basename = os.path.basename(wav).split('.')[0]
        basename = wav.replace('/', '_').replace('.', '_')
        save_path = os.path.join(self.tmp_dir, basename+".csv")
        cmd = 'SMILExtract -C {}/config/ComParE_2016.conf \
            -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1 \
            -I {} -lldcsvoutput {} -instname xx -O ? -noconsoleoutput 1'
        p = subprocess.Popen([cmd.format(self.opensmile_tool_dir, wav, save_path)], stderr=subprocess.PIPE, shell=True)
        err = p.stderr.read()
        if err:
            raise RuntimeError(err)
        
        df = pd.read_csv(save_path, delimiter=';')
        wav_data = np.array(df.iloc[:, 2:])
        if self.downsample > 0:
            if len(wav_data) > self.downsample:
                wav_data = spsig.resample_poly(wav_data, up=1, down=self.downsample, axis=0)
                if self.no_tmp:
                    os.remove(save_path) 
            else:
                raise ValueError('Error in {wav}, signal length must be longer than downsample parameter')

        return wav_data


def make_or_exists(path):
    if not osp.exists(path):
        os.makedirs(path)

def padding_to_fixlen(feat, max_len):
    assert feat.ndim == 2
    if feat.shape[0] >= max_len:
        feat = feat[:max_len]
    else:
        feat = np.concatenate([feat, \
            np.zeros((max_len-feat.shape[0], feat.shape[1]))], axis=0)
    return feat

def comparE_extrct_func(utt_id, extractor, config):
    set_name = utt_id.split('_')[0]
    utt_name = utt_id[utt_id.find('_')+1:]
    wav_path = osp.join(config['audio_dir'], 'audio', set_name, utt_name + '.wav')
    save_dir = osp.join(config['output_dir'], 'comparE')
    if not osp.exists(wav_path):
        feat = np.zeros([1, 130])
        print(f'Error in {wav_path}')
    else:
        feat = extractor(wav_path)
    feat = feat[:1500]
    np.save(osp.join(save_dir, utt_id + '.npy'), feat)

def get_int2name_label(config, set_name):
    # trn val tst 的int2name不唯一
    int2name_path = osp.join(config['target_dir'], set_name, 'int2name.npy')
    label_path = osp.join(config['target_dir'], set_name, 'label.npy')
    int2name = np.load(int2name_path)
    int2name = ["dia"+x[0].split('_')[0] + '_' + "utt"+x[0].split('_')[1] for x in int2name]
    label = np.load(label_path)
    return int2name, label

def make_comparE(config):
    # record = {}
    # mean_std = np.load('/data6/lrc/MELD/feature/mean_std.npz')
    # mean = mean_std['mean']
    # std = mean_std['std']
    # std[std==0.0] = 1.0

    for set_name in ['train', 'dev', 'test']:
        int2name, _ = get_int2name_label(config, set_name)
        int2name = [set_name + '_' + x for x in int2name]
        save_dir = osp.join(config['output_dir'], 'comparE')
        make_or_exists(save_dir)
        extractor = ComParEExtractor(downsample=-1)
        func = partial(comparE_extrct_func, extractor=extractor, config=config)
        pool = mp.Pool(24)
        ret_ft = list(tqdm(pool.imap(func, int2name), total=len(int2name), desc=set_name))
        print(len(ret_ft))
        # ret_ft = [func(x) for x in int2name]
        # ret_ft = np.array(ret_ft)
        # record[set_name] = ret_ft
        # print(ret_ft.shape)
        # np.save(osp.join(save_dir, f'{set_name}.npy'), ret_ft)
    
def calc_mean_std(config):
    save_dir = osp.join(config['output_dir'], 'comparE')
    all_ft_path = glob.glob(osp.join(save_dir, '*.npy'))
    all_ft = []
    for x in tqdm(all_ft_path):
        all_ft.append(np.load(x))
    # all_ft = [np.load(x) for x in tqdm(all_ft_path)]
    all_ft = np.concatenate(all_ft, axis=0)
    print(all_ft.shape)
    mean = np.mean(all_ft, axis=0)
    std = np.mean(all_ft, axis=0)
    print(mean.shape)
    print(std.shape)
    np.savez(osp.join(save_dir, f'mean_std.npz'), mean=mean, std=std)

def normlize(config):
    save_dir = osp.join(config['output_dir'], 'comparE')
    mean_std = np.load(osp.join(save_dir, f'mean_std.npz'))
    mean = mean_std['mean']
    std = mean_std['std']
    std[std==0.0] = 1.0

    for set_name in ['train', 'dev', 'test']:
        int2name, _ = get_int2name_label(config, set_name)
        int2name = [set_name + '_' + x for x in int2name]
        src_dir = osp.join(config['output_dir'], 'comparE')
        tgt_dir = osp.join(config['output_dir'], 'comparE_norm')
        make_or_exists(tgt_dir)
        for utt_id in tqdm(int2name):
            ft = np.load(osp.join(src_dir, utt_id + '.npy'))
            ft = (ft-mean)/std
            save_path = osp.join(tgt_dir, utt_id + '.npy')
            np.save(save_path, ft)

if __name__ == '__main__':
    make_comparE(CONFIG)
    calc_mean_std(CONFIG)
    normlize(CONFIG)
