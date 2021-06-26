import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    ''' one directional LSTM encoder
    '''
    def __init__(self, input_size, hidden_size, embd_method='last'):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size #输入数据的特征维数，通常就是embedding_dim(词向量的维度)
        self.hidden_size = hidden_size #LSTM中隐层的维度
        #batch_first：如果设为True，则输入的数据shape=(batch_size,seq_length卷积后的维度,embedding_dim),
        #             默认是False，即输入的数据shape=(seq_length卷积后的维度,batch_size,embedding_dim)
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        assert embd_method in ['maxpool', 'attention', 'last']
        #'maxpool'就是在时间维度上做maxpooling, 'last'是取最后一个时刻, 'attention'是用attention做加权平均, 通常是maxpool效果最好
        self.embd_method = embd_method
        
        if self.embd_method == 'attention':
            self.attention_vector_weight = nn.Parameter(torch.Tensor(hidden_size, 1))
            self.attention_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )
            self.softmax = nn.Softmax(dim=-1)

    def embd_attention(self, r_out, h_n):
        ''''
        参考这篇博客的实现:
        https://blog.csdn.net/dendi_hust/article/details/94435919
        https://blog.csdn.net/fkyyly/article/details/82501126
        论文：Hierarchical Attention Networks for Document Classification
        formulation:  lstm_output*softmax(u * tanh(W*lstm_output + Bias)
        W and Bias 是映射函数，其中 Bias 可加可不加
        u 是 attention vector 大小等于 hidden size
        '''
        hidden_reps = self.attention_layer(r_out)                       # [batch_size, seq_len, hidden_size]
        atten_weight = (hidden_reps @ self.attention_vector_weight)              # [batch_size, seq_len, 1]
        atten_weight = self.softmax(atten_weight)                       # [batch_size, seq_len, 1]
        # [batch_size, seq_len, hidden_size] * [batch_size, seq_len, 1]  =  [batch_size, seq_len, hidden_size]
        sentence_vector = torch.sum(r_out * atten_weight, dim=1)       # [batch_size, hidden_size]
        return sentence_vector

    def embd_maxpool(self, r_out, h_n):
        # embd = self.maxpool(r_out.transpose(1,2))   # r_out.size()=>[batch_size, seq_len, hidden_size]
                                                    # r_out.transpose(1, 2) => [batch_size, hidden_size, seq_len]
        in_feat = r_out.transpose(1,2)
        embd = F.max_pool1d(in_feat, in_feat.size(2), in_feat.size(2)) #前3个参数：input, kernel_size, stride
        return embd.squeeze(dim=-1) #squeeze：移除数组中维度为1的维度

    def embd_last(self, r_out, h_n):
        #Just for  one layer and single direction
        return h_n.squeeze()

    def forward(self, x):
        '''
        r_out shape: seq_len, batch, num_directions * hidden_size
        hn and hc shape: num_layers * num_directions, batch, hidden_size
        '''
        r_out, (h_n, h_c) = self.rnn(x)
        embd = getattr(self, 'embd_'+self.embd_method)(r_out, h_n) #传递函数的名字
        return embd


if __name__ == '__main__':
    
    #a = LSTMEncoder(342, 128, 'attention')
    a = LSTMEncoder(256, 128, embd_method='maxpool', bidirection=True)
    print(a)
    #data = torch.Tensor(12, 20, 342) #(batch_size, seq_len, embedding_dim(input_size))
    data = torch.Tensor(12, 67, 256) #(batch_size, seq_len, embedding_dim(input_size))
    print(a(data).shape) #这里输出是([12, 128])，(batch_size, hidden_size)
    #hidden_size：每个cell里神经元的个数

    #rnn = nn.LSTM(342, 128, batch_first=True, bidirectional=True)
    rnn = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
    r_out, (h_n, h_c) = rnn(data)
    print('--------------------')
    print(r_out.shape)
    print(h_n.shape)
    print(h_c.shape)