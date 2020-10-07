# -*- coding: utf-8 -*-
"""
   Author :       lirenqiang
   date：          2019/9/25
"""
import torch.utils.data as data
from utils.tools.logger import Logger as logger
from utils.config import cfg
import os
import shutil
import math
import time
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing

from utils.tools.util import progress_monitor
            
class BaseSubjectSampler(object):
    
    def __init__(self, subject):
        self.subject = subject
    
    def sample_one(self):
        pass
    
    def sample_k(self, k):
        pass

class BaseDataSamper(object):
    
    def __init__(self, subjects, sample_k, Sampler):
        self.subjects = subjects
        self.sample_k = sample_k
        
        self.ex = ProcessPoolExecutor()
        self.objs = []
        subject_num = len(self.subjects)
        step = int(math.ceil((subject_num / 20)))
        monitor = progress_monitor(total=len(range(0, subject_num, step)))
        
        for i in range(0, subject_num, step):
            future  = self.ex.submit(self.sample_subjects, self.subjects[i:i+step], self.sample_k, Sampler, self.build_params())
            future.add_done_callback(fn=monitor)
            self.objs.append(future)
        print('data processing in async ...')
        
    def build_params(self):
        return {}
    
    @staticmethod
    def sample_subjects(subjects, sample_num, Sampler, params):
        print('Base DataSampler')
        
    def __len__(self):
        return self.sample_k * len(self.subjects)
    
class MemoryDatasetSampler(BaseDataSamper):
    
    def __init__(self, subjects, sample_k, Sampler):
        super(MemoryDatasetSampler, self).__init__(subjects, sample_k, Sampler)
        
        self.data_lst = None
        
    @staticmethod
    def sample_subjects(subjects, sample_num, Sampler, params):
        
        data_lst = []
        for subject in subjects:
            ss = Sampler(subject)
            data_lst.extend(ss.sample_k(sample_num))
            
        return data_lst
     
    def get_data(self):
        if not self.data_lst:
            self.ex.shutdown(wait=True)
            self.data_lst = []
            for obj in self.objs:
                self.data_lst.extend(obj.result())
        return self.data_lst
    
    def __len__(self):
        return len(self.get_data())
        
        
class FileDatasetSampler(BaseDataSamper):
    
    def __init__(self, subjects, sample_k, Sampler, cache_num=512):
        
        manager = multiprocessing.Manager()
        self.queue = manager.Queue(maxsize=cache_num)
        self.Sampler = Sampler
        self.tmp_folder = os.path.join('/tmp', 'DATA_SAMPLE' + '_' + str(time.time()))
        os.mkdir(self.tmp_folder)
        
        super(FileDatasetSampler, self).__init__(subjects, sample_k, Sampler)
        
        self.tmp_files = manager.Queue()
        self.delete_thread_num = 3
        for i in range(self.delete_thread_num):
            self.ex.submit(self.delete_files, self.tmp_files)
                       
        #for i in range(self.delete_thread_num):
        #    t = threading.Thread(target=self.delete_files, args=(self.tmp_files,))
        #    t.daemon = True
        #    t.start()
        #self.delete_thread = threading.Thread(target=self.delete_files, args=(self.tmp_files,))
        #self.delete_thread.daemon = True
        #self.delete_thread.start()
        
    def build_params(self):
        return {'queue': self.queue,
                  'sampler': self.Sampler,
                  'tempfolder': self.tmp_folder}
    
    @staticmethod
    def delete_files(queue):
        while True:
            pth = queue.get()
            queue.task_done()
            if pth == 'END':
                break
            os.system('rm %s' % pth)
        
    @staticmethod
    def sample_subjects(subjects, sample_num, Sampler, params):
        queue, tmp_folder = params['queue'], params['tempfolder']

        for subject in subjects:
            try:
                ss = Sampler(subject)
            except Exception as ex:
                print('sample %s error %s' % (subject, ex))
                continue
            
            for i in range(sample_num):
                try:
                    data = ss.sample_one()
                except Exception as ex:
                    print('sample %s error %s' % (subject, ex))
                    continue
                    
                while True:
                    try:
                        pth = os.path.join(tmp_folder, '%s_%04d.npy' % (subject, i))
                        np.save(pth, data)
                    except Exception as ex:
                        print('save file error %s' % ex)
                        os.system('rm %s' % pth)
                        time.sleep(1)
                        continue
                        
                    queue.put(pth)
                    break            
    
    def __len__(self):
        return self.sample_k * len(self.subjects)
    
    def get_data(self):
        
        pth = self.queue.get()
        data = np.load(pth, allow_pickle=True).item()
        self.queue.task_done()
       
        self.tmp_files.put(pth)
        
        return data
    
    def __del__(self):
        for i in range(self.delete_thread_num):
            self.tmp_files.put('END')
        self.ex.shutdown(wait=True)
        self.tmp_files.join()
        if os.path.exists(self.tmp_folder):
            shutil.rmtree(self.tmp_folder)

class BaseDataset(data.Dataset):

    def __init__(self, para_dict, stage):
        self.stage = stage
        assert self.stage in ("train", "val", "test"), "Unsupported stage:{}".format(stage)
        self.para_dict = para_dict
        self.logger = logger
        self.num = None
        if self.stage == "train":
            self.train_init()
        elif self.stage == "val":
            self.val_init()
        elif self.stage == "test":
            self.test_init()
        else:
            raise Exception("Unknown stage:{}".format(self.stage))
        self.logger.info("stage:{} load:{} nums!".format(self.stage, len(self)))
    
    def len(self):
        assert self.num, "You must init self.num in {}, which is the data nums".format(self.stage)
        if self.stage == "train":
            if "TRAIN_NUM_USE" in cfg.TRAIN.DATA.keys():
                if cfg.TRAIN.DATA.TRAIN_NUM_USE > 0:
                    self.num = cfg.TRAIN.DATA.TRAIN_NUM_USE
        elif self.stage == "val":
            if "VAL_NUM_USE" in cfg.TRAIN.DATA.keys():
                if cfg.TRAIN.DATA.VAL_NUM_USE > 0:
                    self.num = cfg.TRAIN.DATA.VAL_NUM_USE
        elif self.stage == "test":
            if "TEST_NUM_USE" in cfg.TEST.DATA.keys():
                if cfg.TEST.DATA.TEST_NUM_USE > 0:
                    self.num = cfg.TEST.DATA.TEST_NUM_USE
        return self.num
    
    def __len__(self):
        return self.len()

    def __getitem__(self, index):
        if self.stage == "train":
            return self.train_load(index)
        elif self.stage == "val":
            return self.val_load(index)
        elif self.stage == "test":
            return self.test_load(index)
        else:
            raise Exception("Unknown stage:{}".format(self.stage))

    def train_init(self):
        raise NotImplementedError

    def val_init(self):
        raise NotImplementedError

    def test_init(self):
        raise NotImplementedError

    def train_load(self, index):
        raise NotImplementedError

    def test_load(self, index):
        raise NotImplementedError

    def val_load(self, index):
        raise NotImplementedError

class RandomSequenceSampler(data.sampler.Sampler):
    
    """
    load the mini-batch from the same subject in the same worker, so cache can be used
    
    Args:
        subject_num: the number of trainning subjects
        sample_num_per_subject: the number of samples for each subject
        batch_size: size of mini-batch
        num_workers: how many subprocesses to use for data loading
        batch_from_multiple_subject: if set true, then samples from multiple subjects in a mini-batch
        max_subject_in_batch: if None, set it to batch_size as default
    """
    
    def __init__(self, subject_num, sample_num_per_subject, batch_size, num_workers=2, batch_from_multiple_subject=True, max_subject_in_batch=None):
        
        assert sample_num_per_subject % batch_size == 0, 'sample_num_per_subject % batch_size != 0'
        
        self.subject_num = subject_num
        self.batch_size = batch_size
        self.sample_num_per_subject = sample_num_per_subject
        self.num_workers = num_workers
        self.total_num = subject_num * sample_num_per_subject
        self.batch_from_multiple_subject = batch_from_multiple_subject
        self.max_subject_in_batch = batch_size if max_subject_in_batch is None else max_subject_in_batch
        
    def get_indices(self):
 
        n, k, b, w = self.subject_num, self.sample_num_per_subject, self.batch_size, self.num_workers
        ss = list(range(n))
        random.shuffle(ss)
        indices = []
        
        pp, nn = list(range(w)), [0] * w
        #first step, calculate each worker's sample indices
        a = [[] for i in range(w)]
        j = 0
        while np.min(pp) < n:
            for i in range(w):
                if pp[i] >= n:
                    continue        
                if nn[i]  < k:
                    a[j % 4].extend([ss[pp[i]] * k + nn[i] + j for j in range(b)])
                    nn[i] += b
                    j += 1            
                    if nn[i] >= k:
                        pp[i] = pp[i] + w
                        nn[i] = 0
        
        #second step, shuffle mini-batch
        if self.batch_from_multiple_subject:
            a2 = [[] for i in range(w)]
            for i in range(w):
                j = 0
                while j < len(a[i]):
                    nn = random.randint(2, self.max_subject_in_batch) * k
                    bb = a[i][j:j+nn].copy()
                    random.shuffle(bb)
                    a2[i].extend(bb)
                    j = j + nn
            a = a2
       
        #third step, get the final indices
        pp = [0] * w
        while np.sum(pp) < n*k:
            for i in range(w):
                if pp[i] >= len(a[i]):
                    continue
                indices.extend(a[i][pp[i]:pp[i]+b])
                pp[i] = pp[i] + b
        
        return indices
    
    def __iter__(self):
        
        indices = self.get_indices()
        return iter(indices)
    
    def __len__(self):
        return self.total_num