import io
import os
import json
from tqdm import tqdm
from collections import defaultdict
import lmdb
from lz4.frame import compress, decompress
import torch
import numpy as np 
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

def load_txt_db(db_dir):
    # db loading
    env_in = lmdb.open(db_dir)
    txn_in = env_in.begin()
    db = {}
    for key, value in txn_in.cursor():
        db[key] = value
    print('db length:', len(db)) # db length: 443757
    env_in.close()
    return db

def load_img_db(img_dir, conf_th=0.2, max_bb=100, min_bb=10, num_bb=36, compress=False):
    if conf_th == -1:
        db_name = f'feat_numbb{num_bb}'
        name2nbb = defaultdict(lambda: num_bb)
    else:
        db_name = f'feat_th{conf_th}_max{max_bb}_min{min_bb}'
        nbb = f'nbb_th{conf_th}_max{max_bb}_min{min_bb}.json'
        if not os.path.exists(f'{img_dir}/{nbb}'):
            # nbb is not pre-computed
            name2nbb = None
        else:
            name2nbb = json.load(open(f'{img_dir}/{nbb}'))
            # => {'coco_test2015_000000043222.npz': 57, ...}
    if compress:
        db_name += '_compressed'
    if name2nbb is None:
        if compress:
            db_name = 'all_compressed'
        else:
            db_name = 'all'
    
    # db loading
    env = lmdb.open(f'{img_dir}/{db_name}', readonly=True, create=False)
    txn = env.begin(buffers=True)
    
    return name2nbb, txn

def load_single_img(txn, key, compress=False):
    # load single image with its key
    if isinstance(key, str):
        key = key.encode('utf-8')
    dump = txn.get(key)
    if compress:
        with io.BytesIO(dump) as reader:
            img_dump = np.load(reader, allow_pickle=True)
            img_dump = {'features': img_dump['features'],
                        'norm_bb': img_dump['norm_bb']}
    else:
        img_dump = msgpack.loads(dump, raw=False)
    return img_dump

def load_single_region(txn, key, compress=False):
    _, img_key, bb_idx = split_region_key(key)
    img_dump = load_single_img(txn, img_key, compress=compress)
    return img_dump['features'][bb_idx]
    
def load_single_txt(data):
    return msgpack.loads(decompress(data), raw=False)
