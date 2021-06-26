import torch
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.cnn import EncCNN1d, ResBlock
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier

class CNN_LSTMEncoder(nn.Module):
    def __init__(self, input_dim, enc_channel, embd_size, embd_method, cnn_type):
        super().__init__()
        assert(cnn_type in ['EncCNN1d', 'ResBlock'])
        #CNN层
        if cnn_type == 'EncCNN1d':
            self.cnn = EncCNN1d(input_dim, enc_channel)
        else:
            self.cnn = ResBlock(input_dim, enc_channel)
        #lstm层
        self.lstm = LSTMEncoder(enc_channel, embd_size, embd_method=embd_method)
        self.module = nn.Sequential(self.cnn, self.lstm)

    def forward(self, x):
        feat = self.module(x)
        return feat


class AddFusionModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--use_downsample_a', action='store_true', help='if specified, use downsample. else, use CNN.') #hzp add
        parser.add_argument('--cnn_a', type=str, default='EncCNN1d', choices=['EncCNN1d', 'ResBlock'], \
            help='which CNN in audio: EncCNN1d, ResBlock. Need it when use CNN') #hzp add
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--enc_channel_a', type=int, default=128, help='acoustic encoder channel, need it when use CNN') #hzp add
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='lexical input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='visual embedding method,last,mean or atten')
        parser.add_argument('--cls_layers', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--modality', type=str, help='which modality to use for model')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.loss_names = ['CE']
        self.modality = opt.modality
        self.model_names = ['C']
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))

        assert(opt.embd_size_a == opt.embd_size_v and opt.embd_size_a == opt.embd_size_l)#
        cls_input_size = opt.embd_size_a#

        self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        # acoustic model
        if 'A' in self.modality:
            self.model_names.append('A')
            if opt.use_downsample_a: #使用降采样作为去噪方法
                self.netA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
            else: #使用CNN作为去噪方法
                self.netA = CNN_LSTMEncoder(opt.input_dim_a, opt.enc_channel_a, opt.embd_size_a, opt.embd_method_a, opt.cnn_a)
            
        # lexical model
        if 'L' in self.modality:
            self.model_names.append('L')
            self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l)

        # visual model
        if 'V' in self.modality:
            self.model_names.append('V')
            self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
            
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        if 'A' in self.modality:
            self.acoustic = input['A_feat'].float().to(self.device)
        if 'L' in self.modality:
            self.lexical = input['L_feat'].float().to(self.device)
        if 'V' in self.modality:
            self.visual = input['V_feat'].float().to(self.device)
        
        self.label = input['label'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        final_embd = []
        if 'A' in self.modality:
            self.feat_A = self.netA(self.acoustic)
            final_embd.append(self.feat_A)

        if 'L' in self.modality:
            self.feat_L = self.netL(self.lexical)
            final_embd.append(self.feat_L)
        
        if 'V' in self.modality:
            self.feat_V = self.netV(self.visual)
            final_embd.append(self.feat_V)
        
        # get model outputs
        self.add_fusion = final_embd[0]
        for embd_item in final_embd[1:]:
            self.add_fusion = self.add_fusion + embd_item
        self.logits, self.ef_fusion_feat = self.netC(self.add_fusion)
        self.pred = F.softmax(self.logits, dim=-1)
        
    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_CE = self.criterion_ce(self.logits, self.label) #torch.nn.CrossEntropyLoss()中已经包含了softmax的过程，所以这里的输入是self.logits
        loss = self.loss_CE
        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 0.1) # 0.1

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        # backward
        self.optimizer.zero_grad()  
        self.backward()       
        self.optimizer.step() 
