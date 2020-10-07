
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import time

class diceEval:
    
    def __init__(self, nClasses, per_image=True):

        self.per_image = per_image
        self.n_class = nClasses
        self.reset()
    
    def reset(self):
        
        self.total_dice = 0
        self.total_num = 0
        self.results = np.zeros(self.n_class)
        
        #dice = torch.zeros([self.nClasses])
        #self.dice = dice.cuda()
    '''
    def addBatch(self, predict, gt):
        batch_size = predict.size(0)
        dice = torch.zeros([self.nClasses])
        
        tt = time.time()
        #dice = dice.cuda()
        self.dice.zero_()
        
        print ('t: %.4f' % (time.time() - tt))
        for i in range(1, self.nClasses):
            
            tt = time.time()
            input_i = (predict == i)
            target_i = (gt == i)
            print ('t00: %.4f' % (time.time() - tt))
            
            tt = time.time()
            num = (input_i * target_i)
            num = torch.sum(num)
            print ('t01: %.4f' % (time.time() - tt))
            
            tt = time.time()
            den1 = torch.sum(input_i)
            den2 = torch.sum(target_i)
            print ('t02: %.4f' % (time.time() - tt))
            
            epsilon = 1e-6
            tt = time.time()
            self.dice[i] = (2 * num + epsilon) / (den1 + den2 + epsilon)
            print ('t03: %.4f' % (time.time() - tt))
        
        dice = torch.sum(self.dice[1:])
        dice = dice / (self.nClasses - 1)
        self.total_dice += dice
        self.total_num += batch_size
        
    
    def addBatch(self, predict, gt):
        batch_size = predict.size(0)
        dice = torch.zeros([predict.size(0), self.nClasses])
        
        dice = dice.to(torch.device('cuda'))
        
        for i in range(1, self.nClasses):
            
            input_i = (predict == i).float().view(batch_size, -1)
            target_i = (gt == i).float().view(batch_size, -1)
            
            tt = time.time()
            num = (input_i * target_i)
            num = torch.sum(num, dim=1)
            
            den1 = torch.sum(input_i, dim=1)
            den2 = torch.sum(target_i, dim=1)
            
            epsilon = 1e-6
            
            dice[:, i] = (2 * num + epsilon) / (den1 + den2 + epsilon)
        
        dice = dice[:, 1:]
        dice = torch.sum(dice, dim=1)
        dice = dice / (self.nClasses - 1)
        self.total_dice += torch.sum(dice).item()
        self.total_num += batch_size
    '''
    
    def compute_dice(self, pred, gt): 
        pred, gt = pred.view(-1), gt.view(-1)
        num = torch.sum(pred * gt)
        den1 = torch.sum(pred)
        den2 = torch.sum(gt)
        epsilon = 1e-4
        dice = (2 * num + epsilon) / (den1 + den2 + epsilon)
        return dice.item()
    
    def compute(self, pred, gt):         
        '''for each segmentation result, calculate dice score, and last item:*[-1] store binary class dice value'''
        results = np.zeros(self.n_class)
        for i in range(1, self.n_class):
            results[i-1] = self.compute_dice((pred == i).float(), (gt == i).float())
        
        if self.n_class == 2:
            results[-1] = results[0]
        else:
            results[-1] = self.compute_dice((pred > 0).float(), (gt > 0).float())
        
        return results 
        
    
    def addBatch(self, predict, gt):        
        if self.per_image:
            for p, q in zip(predict, gt):
                self.results += self.compute(p, q)
                self.total_num += 1                
        else:
            self.results += self.compute(predict, gt)
            self.total_num += 1

    def getMetric(self):
        
        #epsilon = 1e-8
        #return self.total_dice / (self.total_num + epsilon)
        
        if self.total_num == 0:
            return self.results
        else:
            return self.results / self.total_num