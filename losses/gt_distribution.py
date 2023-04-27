from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

# groud truth Laplace distribution
def LaplaceDisp2Prob(Gt,maxdisp=192,m=3,n=3):
    N,H,W = Gt.shape
    b = 0.8
            
    Gt = torch.unsqueeze(Gt,1)
    disp = torch.arange(maxdisp,device=Gt.device)
    disp = disp.reshape(1,maxdisp,1,1).repeat(N,1,H,W)
    cost = -torch.abs(disp-Gt) / b

    return F.softmax(cost,dim=1)




def adaptive_multi_modal_CrossEntropy_loss(x,disp,mask,maxdisp,m=1,n=9):
    N,H,W = disp.shape
    
    num = mask.sum()
    x = torch.log(x + 1e-30)
    mask = torch.unsqueeze(mask,1).repeat(1,maxdisp,1,1)
    
    Gt1 = LaplaceDisp2Prob(disp,maxdisp)
    
    Gtpad = F.pad(disp,(int(n/2-0.5),int(n/2+0.5),int(m/2-0.5),int(m/2+0.5)),mode='replicate')
    Gt_list = torch.zeros(N,m*n,H,W,device=x.device)
    for i in range(m):
        for j in range(n):
            Gt_list[:,n*i+j,:,:] = Gtpad[:,i:-(m-i),j:-(n-j)]
    invalid_num = torch.sum((Gt_list == 0),dim=1,keepdim=True)
    mean = torch.mean(Gt_list,dim=1,keepdim=True) * m*n / (m*n - invalid_num).clamp(min=1)
    valid = (torch.abs(mean-disp.unsqueeze(1)) > 5) #N1HW

    index = ((Gt_list < (disp.unsqueeze(1)-5)) + (Gt_list > (disp.unsqueeze(1)+5))) * (Gt_list > 0)
    Gt_list = Gt_list * index

    mean = torch.mean(Gt_list,dim=1) * (m*n) / torch.sum(index,dim=1).clamp(min=1)
    Gt2 = LaplaceDisp2Prob(mean,maxdisp)
    
    w = torch.sum(index,dim=1,keepdim=True) * 0.2 / (m*n - 1 - invalid_num).clamp(min=1)

    Gt = ((1-w*valid)*Gt1 + (w*valid)*Gt2).detach_()
    loss =  - (Gt[mask]*x[mask]).sum() / num 

    return loss