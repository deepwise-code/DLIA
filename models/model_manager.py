

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.config import cfg
    
class ModelManager(object):
    
    def __call__(self):
        raise NameError('Unknown Task Type')
        # if cfg.TASK.TYPE == 0:
        #     from models.cls.model_manager import _ModelManager
        # elif cfg.TASK.TYPE == 1:
        #     from models.seg.model_manager import _ModelManager
        # else:
        #     raise NameError('Unknown Task Type')
        
        # return _ModelManager()()
        