
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

def mkdir_safe(d):
    """
    Make Multi-Directories safety and thread friendly.
    """
    sub_dirs = d.split('/')
    cur_dir = ''
    max_check_times = 5
    sleep_seconds_per_check = 0.001
    for i in range(len(sub_dirs)):
        cur_dir += sub_dirs[i] + '/'
        for check_iter in range(max_check_times):
            if not os.path.exists(cur_dir):
                try:
                    os.mkdir(cur_dir)
                except Exception as e:
                    #print '[WARNING] ', str(e)
                    time.sleep(sleep_seconds_per_check)
                    continue
            else:
                break
