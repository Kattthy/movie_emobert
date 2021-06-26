import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import h5py
import os
import numpy as np
from tqdm import tqdm
import glob
import csv
import json

CONFIG = {
    "text_dir": "/data8/hzp/datasets/MELD_process/text",
    "data_dir": "/data8/hzp/datasets/MELD.Raw",
    "target_dir": "/data8/hzp/datasets/MELD_target",
    "output_dir": "/data8/hzp/datasets/MELD_feature"
}

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

def get_trn_val_tst(target_root_dir, setname):
    int2name = np.load(os.path.join(target_root_dir, '{}_int2name.npy'.format(setname)))
    int2label = np.load(os.path.join(target_root_dir, '{}_label.npy'.format(setname)))
    assert len(int2name) == len(int2label)
    return int2name, int2label

class BertExtractor(object):
    def __init__(self, cuda=False, cuda_num=None):
        self.tokenizer = BertTokenizer.from_pretrained('/data2/lrc/bert_cache/pytorch')
        self.model = BertModel.from_pretrained('/data2/lrc/bert_cache/pytorch')
        self.model.eval()

        if cuda:
            self.cuda = True
            self.cuda_num = cuda_num
            self.model = self.model.cuda(self.cuda_num)
        else:
            self.cuda = False
        
    def tokenize(self, word_lst):
        word_lst = ['[CLS]'] + word_lst + ['[SEP]']
        word_idx = []
        ids = []
        for idx, word in enumerate(word_lst):
            ws = self.tokenizer.tokenize(word)
            if not ws:
                # some special char
                continue
            token_ids = self.tokenizer.convert_tokens_to_ids(ws)
            ids.extend(token_ids)
            if word not in ['[CLS]', '[SEP]']:
                word_idx += [idx-1] * len(token_ids)
        return ids, word_idx
    
    def get_embd(self, token_ids):
        # token_ids = torch.tensor(token_ids)
        # print('TOKENIZER:', [self.tokenizer._convert_id_to_token(_id) for _id in token_ids])
        token_ids = torch.tensor(token_ids).unsqueeze(0)
        if self.cuda:
            token_ids = token_ids.to(self.cuda_num)
            
        with torch.no_grad():
            outputs = self.model(token_ids)
            
            # last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        return sequence_output, pooled_output

    def extract(self, text):
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
        if self.cuda:
            input_ids = input_ids.cuda(self.cuda_num)

        with torch.no_grad():
            
            #有个别句的字幕有些错误，是一堆数字和字母（主要就是m, b, l）并以空格分隔。
            #如果将他们看作单词则这一句里可能会有很多单词，在tokenize为sub-word后可能会超过512个。
            #这时候模型可能就会报错（看错误信息模型允许的输入最大长度是512），
            #所以对于过长的sub-word序列要直接截断（反正正常的句子不会这么长的）
            if len(input_ids[0]) > 512: #input_ids：[[id, id, ...]]
                input_ids = input_ids.squeeze()
                input_ids = input_ids[:512]
                input_ids = input_ids.unsqueeze(dim=0)

            outputs = self.model(input_ids)
            
            # last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        return sequence_output, pooled_output
        #sequence_output：encoder端最后一层编码层的特征向量
        #pooled_output：[CLS]这个token对应的向量，把它作为整个句子的特征向量


def make_all_bert(config):
    '''
    trn cal tst 里的utt_id不唯一！
    '''
    extractor = BertExtractor(cuda=True, cuda_num=0)
    text_dir = config['text_dir']
    text_files = glob.glob(text_dir + '/*.tsv')
    utt_id2transcript = {}
    setname2set = {'train':'trn','test':'tst','dev':'val'}
    for text_file in text_files:
        setname = os.path.basename(text_file).split('.')[0]
        set_ = setname2set[setname]
        utt_id2transcript[set_] = {}
        with open(text_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                utt_id = line[0]
                transcript = line[1]
                utt_id2transcript[set_][utt_id] = transcript
    
    trn_int2name, _ = get_trn_val_tst(config['target_dir'], 'trn')
    val_int2name, _ = get_trn_val_tst(config['target_dir'], 'val')
    tst_int2name, _ = get_trn_val_tst(config['target_dir'], 'tst')
    trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
    val_int2name = list(map(lambda x: x[0].decode(), val_int2name))
    tst_int2name = list(map(lambda x: x[0].decode(), tst_int2name))
    #all_utt_ids = trn_int2name + val_int2name + tst_int2name
    mkdir(os.path.join(config['output_dir'], 'L', 'bert'))
    trn_h5f = h5py.File(os.path.join(config['output_dir'], 'L', 'bert', 'trn.h5'), 'w')
    tst_h5f = h5py.File(os.path.join(config['output_dir'], 'L', 'bert', 'tst.h5'), 'w')
    val_h5f = h5py.File(os.path.join(config['output_dir'], 'L', 'bert', 'val.h5'), 'w')
    for utt_id in trn_int2name:
        sequence_feat, pooled_feat = extractor.extract(utt_id2transcript['trn'][utt_id])
        sequence_feat = sequence_feat.cpu()
        trn_h5f[utt_id] = sequence_feat.squeeze()
    for utt_id in tst_int2name:
        sequence_feat, pooled_feat = extractor.extract(utt_id2transcript['tst'][utt_id])
        sequence_feat = sequence_feat.cpu()
        tst_h5f[utt_id] = sequence_feat.squeeze()
    for utt_id in val_int2name:
        sequence_feat, pooled_feat = extractor.extract(utt_id2transcript['val'][utt_id])
        sequence_feat = sequence_feat.cpu()
        val_h5f[utt_id] = sequence_feat.squeeze()


if __name__ == '__main__':
    '''
    pwd = os.path.abspath(__file__) #获取当前文件的绝对路径
    pwd = os.path.dirname(pwd) #获取该文件的上级目录
    pwd = os.path.dirname(pwd) #获取该文件的上级目录
    config_path = os.path.join(pwd, '../', 'data/config', 'MELD_config.json')
    config = json.load(open(config_path))
    '''
    make_all_bert(CONFIG)