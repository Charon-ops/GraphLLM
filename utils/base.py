import torch
import json
import os
import random
import numpy as np
import subprocess
import re
import logging

def set_seed(seed: int):
    """set down all random factors for reproducing results in future"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_default_fp(fp: str):
    if fp == 'fp16':
        torch.set_default_dtype(torch.float16)
    elif fp == 'bf16':
        torch.set_default_dtype(torch.bfloat16)
    elif fp == 'fp64':
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

def select_gpu():
    try:
        nvidia_info = subprocess.run('nvidia-smi', stdout=subprocess.PIPE).stdout.decode()
    except UnicodeDecodeError:
        nvidia_info = subprocess.run('nvidia-smi', stdout=subprocess.PIPE).stdout.decode("gbk")
    used_list = re.compile(r"(\d+)MiB\s+/\s+\d+MiB").findall(nvidia_info)
    used = [(idx, int(num)) for idx, num in enumerate(used_list)]
    sorted_used = sorted(used, key=lambda x: x[1])
    print(f'auto select gpu-{sorted_used[0][0]}, sorted_used: {sorted_used}')
    return sorted_used[0][0]

def set_device(gpu) -> str:
    assert gpu < torch.cuda.device_count(), f'gpu {gpu} is not available'
    if not torch.cuda.is_available():
        return 'cpu'
    if gpu == -1:  gpu = select_gpu()
    return f'cuda:{gpu}'


def set_logger(output_dir=None):
    """ set a root logger"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_filename = os.path.join(output_dir, 'process.log') 
    logging.basicConfig(format = '%(asctime)s %(levelname)-8s %(message)s',  # '%(asctime)s - %(levelname)-8s - %(name)s -   %(message)s'
                        datefmt = '%H:%M:%S %m/%d/%Y',
                        level = logging.INFO,
                        filename=log_filename,
                        filemode='w'
                        )
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
def log_hyperparams(args):
    for arg in vars(args):
        logging.info(f'{arg} = {getattr(args, arg)}')