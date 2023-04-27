import torch.nn.functional as F
from losses.gt_distribution import *


__loss__ = {
    "SL1" : F.smooth_l1_loss,
    "ADL" : adaptive_multi_modal_CrossEntropy_loss,
}