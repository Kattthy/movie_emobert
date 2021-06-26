import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityFusionAttn(nn.Module):
    def __init__(self, embd=128):
        super().__init__()
        self.embd = embd
        self.w = nn.Linear(embd, 1)
        self.transform = nn.Sequential(nn.Linear(embd, embd), nn.ReLU())
    
    def forward(self, A, V, L):
        '''
        A, V, L: [batch_size, embed_size]
        '''
        #print('A', A.shape) #torch.Size([bs, 128])
        #print('transform', self.transform(A).shape) #transform torch.Size([bs, 128])
        #print('A_weight', self.w(self.transform(A)).shape)


        A_weight = self.w(self.transform(A)) # [bs, 1]
        V_weight = self.w(self.transform(V)) # [bs, 1]
        L_weight = self.w(self.transform(L)) # [bs, 1]
        weight = torch.cat([A_weight, V_weight, L_weight], dim=1) #[bs, 3]
        weight = F.softmax(weight / math.sqrt(self.embd)) # [bs, 3]

        #print(A.shape)#[bs, 128]
        #print(torch.unsqueeze(weight[:, 0], dim=1).shape)#[bs]

        A = A * torch.unsqueeze(weight[:, 0], dim=1)
        V = V * torch.unsqueeze(weight[:, 1], dim=1)
        L = L * torch.unsqueeze(weight[:, 2], dim=1)
        return A + V + L   # [bs, embd]