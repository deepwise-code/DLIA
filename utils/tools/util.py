# -*- coding=utf-8 -*-
import time
from functools import wraps


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 0 if (self.count < 1e-5) else (self.sum / self.count)


def progress_monitor(total, verbose=1):
    '''状态参数'''
    finished_count = 0
    time_start = time.time()

    def _print(*args, ):
        nonlocal finished_count
        finished_count += 1
        if finished_count % verbose == 0:
            print("\rprogress: %d/%d = %.2f; time: %.2f s" % (
                finished_count, total, finished_count / total, time.time() - time_start), end="")
        if finished_count == total:
            print('\nfinished: %d, total cost time: %.2f s' % (total, time.time() - time_start,))

    return _print


def count_time(func):
    """统计时间的装饰器"""
    @wraps(func)
    def int_time(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        cost_time = end_time - start_time
        print('程序 %s 共计用时: %.2f s' % (func.__name__, cost_time,))
        return res

    return int_time
