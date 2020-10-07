
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

from utils.config import cfg
from utils.tools.logger import Logger as Log

class ModuleHelper(object):
    
    CONV = nn.Conv3d if cfg.MODEL.DIM == '3d' else nn.Conv2d
    DECONV = nn.ConvTranspose3d if cfg.MODEL.DIM == '3d' else nn.ConvTranspose2d
    MAXPOOL = nn.MaxPool3d if cfg.MODEL.DIM == '3d' else nn.MaxPool2d
    AVGPOOL = nn.AvgPool3d if cfg.MODEL.DIM == '3d' else nn.AvgPool2d
    ADAPTIVEAVGPOOL = nn.AdaptiveAvgPool3d if cfg.MODEL.DIM == '3d' else nn.AdaptiveAvgPool2d
    DROPOUT = nn.Dropout3d if cfg.MODEL.DIM == '3d' else nn.Dropout2d
    
    F_AVG_POOL = F.avg_pool3d if cfg.MODEL.DIM == '3d' else F.avg_pool2d
    
    @staticmethod
    def UPSAMPLE(scale_factor, **kwargs):
        return nn.Upsample(scale_factor=scale_factor, 
                           mode='trilinear' if cfg.MODEL.DIM == '3d' else 'bilinear',
                           **kwargs)
    
    @staticmethod
    def BN(planes, **kwargs):
        if cfg.MODEL.DIM == '3d':
            return ModuleHelper.BN3(planes, **kwargs)
        else:
            return ModuleHelper.BN2(planes, **kwargs)
    
    @staticmethod
    def BN3(planes, **kwargs):  
        if cfg.MODEL.BN == 'bn':
            return nn.BatchNorm3d(planes, **kwargs)
        else:
            return nn.Sequential()
   
    @staticmethod
    def BN2(planes, **kwargs):
        if cfg.MODEL.BN == 'bn':
            return nn.BatchNorm2d(planes, **kwargs)
        else:
            return nn.Sequential()
    
    @staticmethod
    def BNReLU(planes, **kwargs):
        
        return nn.Sequential(
            ModuleHelper.BN(planes, **kwargs),
            nn.ReLU()
        )
    
    @staticmethod
    def INTERPOLATE(x, size, **kwargs):
        mode='trilinear' if cfg.MODEL.DIM == '3d' else 'bilinear'
        return F.interpolate(x, size, mode=mode, **kwargs)
    
    @staticmethod
    def load_model(model, pretrained=None, all_match=True):
        if pretrained is None:
            return model
        
        Log.info('Loading pretrained model:{}'.format(pretrained))
        pretrained_dict = torch.load(pretrained)
            
        if all_match:
            model_dict = model.state_dict()
            load_dict = dict()
            for k, v in pretrained_dict.items():
                if 'prefix.{}'.format(k) in model_dict:
                    load_dict['prefix.{}'.format(k)] = v
                else:
                    load_dict[k] = v
            
            model.load_state_dict(load_dict)
        else:
            model_dict = model.state_dict()
            load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            Log.info('Matched Keys: {}'.format(load_dict.keys()))
            model_dict.update(load_dict)
            model.load_state_dict(model_dict)
        
        return model
        
    @staticmethod
    def constant_init(module, val, bias=0):
        nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
    
    @staticmethod
    def xavier_init(module, gain=1, bias=0, distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
    
    @staticmethod
    def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
    
    @staticmethod
    def uniform_init(module, a=0, b=1, bias=0):
        nn.init.uniform_(module.weight, a, b)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
    
    @staticmethod
    def kaiming_init(module,
                     mode='fan_in',
                     nonlinearity='leaky_relu',
                     bias=0,
                     distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
    

CONV = ModuleHelper.CONV
DECONV = ModuleHelper.DECONV
MAXPOOL = ModuleHelper.MAXPOOL
AVGPOOL = ModuleHelper.AVGPOOL
ADAPTIVEAVGPOOL = ModuleHelper.AVGPOOL
BN = ModuleHelper.BN
BNReLU = ModuleHelper.BNReLU
UPSAMPLE = ModuleHelper.UPSAMPLE
INTERPOLATE = ModuleHelper.INTERPOLATE
DROPOUT = ModuleHelper.DROPOUT
F_AVG_POOL = ModuleHelper.F_AVG_POOL