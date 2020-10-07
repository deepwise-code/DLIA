import torch
import numpy as np

def one_hot(gt, n_class):
    if gt.dim() == 3:
        b, w, h = gt.size()
        y = torch.FloatTensor(b, n_class, w, h).zero_()
    elif gt.dim() == 4:
        b, w, h, d = gt.size()
        y = torch.FloatTensor(b, n_class, w, h, d).zero_()
    else:
        return gt
    
    gt = gt.unsqueeze(1)
    y = y.scatter_(1, gt, 1.0)
    return y

def tensor_dice(pred, gt, n_class):
    
    dices = []
    for idx in range(1, n_class):
        
        fg_pred = pred == idx
        fg_gt = gt == idx
        
        fg_union_num = torch.sum(fg_pred).item() + torch.sum(fg_gt).item()
        
        #if fg_union_num.item() == 0:
        if fg_union_num == 0:
            dices.append(1.0)
        else:
            dices.append(2.0 * torch.sum(fg_pred * fg_gt).item() / fg_union_num)
            #dices.append(2.0 * torch.sum(fg_pred * fg_gt).item() / fg_union_num.item())
    
    avg_dice = np.sum(dices) / (n_class - 1)
    
    return avg_dice, dices