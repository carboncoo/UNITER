import os
import sys
import time
import random

import mlflow
from omegaconf import DictConfig, ListConfig, OmegaConf

def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)

def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)
    else:
        mlflow.log_param(f'{parent_name}', element)
          
          
#################### GPU Utils ####################
            
def gpu_info():
    info = os.popen('nvidia-smi|grep %').read().strip().split('\n')

    def single_gpu(info_t):
        power = int(info_t.split('|')[1].strip().split()[-3][:-1])
        memory = int(info_t.split('|')[2].split('/')[0].strip()[:-3])
        return power, memory

    return [single_gpu(info_t) for info_t in info]

def infer_gpu(memory_limit=1000, power_limit=30, interval=2, limited=None, require_n=1, no_wait=False, output_style='raw'):
    def is_available(info):
        return info[0] < power_limit and info[1] < memory_limit
        # return False
    def gpu_status(info):
        return 'power:%d W | memory:%d MiB |' % (info)
    
    gpu_infos = gpu_info()
    if not limited:
        limited = list(range(len(gpu_infos)))

    gpu_available = [gpu_id for gpu_id in limited if is_available(gpu_infos[gpu_id])]
    
    if no_wait:
        if require_n != -1:
            gpu_available = gpu_available[:require_n]
        if output_style == 'raw':
            gpu_available = ','.join(map(str, gpu_available))
        return gpu_available
    
    i = 0
    while True:
        while len(gpu_available) < require_n:
            i = i % 5
            symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '| \n'
            # to_top = ''.join(['\r' * len(limited)])
            gpu_statuses = '\n'.join([gpu_status(gpu_infos[gpu_id]) for gpu_id in limited])

            # import ipdb; ipdb.set_trace()
            # print(to_top + symbol + gpu_statuses)
            sys.stdout.write(symbol + gpu_statuses + '\n')
            sys.stdout.flush()
            time.sleep(interval)
            i += 1

            gpu_infos = gpu_info()
            gpu_available = [gpu_id for gpu_id in limited if is_available(gpu_infos[gpu_id])]
        
        time.sleep(interval * random.randint(1,5))
        gpu_infos = gpu_info()
        gpu_available = [gpu_id for gpu_id in limited if is_available(gpu_infos[gpu_id])]

        if len(gpu_available) >= require_n:
            break
    
    if require_n != -1:
        gpu_available = gpu_available[:require_n]
    if output_style == 'raw':
        gpu_available = ','.join(map(str, gpu_available))
    return gpu_available