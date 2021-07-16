import os
import re
import sys
import lmdb
import json
import torch
import pickle
import random
import msgpack
import itertools 
import numpy as np
import msgpack_numpy
from models import InferSent
from analyze import load_txt_db
from sklearn.cluster import k_means
from transformers import AutoTokenizer
from lz4.frame import compress, decompress
from os.path import exists, abspath, dirname

seed = 2
sample_num = 200000
txt_dir_in = "/data/share/UNITER/ve/txt_db/ve_train.db"
txt_dir_out = "/data/share/UNITER/ve/da/sample/seed%d/txt_db/ve_train.db"%(seed)

def save_db(db_dir, db, meta, id2len, txt2img, img2txts):
    if not exists(db_dir):
            os.makedirs(db_dir)
    else:
        raise ValueError('Found existing DB. Please explicitly remove '
                        'for re-processing')

    write_cnt = 0
    env_out = lmdb.open(db_dir, map_size=int(3e8))
    txn_out = env_out.begin(write=True)
    for k, v in db.items():
        txn_out.put(k, v)
        write_cnt += 1
        if write_cnt % 1000 == 0:
            txn_out.commit()
            txn_out = env_out.begin(write=True)
            print('write %d'%(write_cnt))
    txn_out.commit()
    env_out.close()
    print('write count: ', write_cnt)

    json.dump(meta, open(txt_dir_out + '/meta.json', 'w'))
    json.dump(id2len, open(txt_dir_out + '/id2len.json', 'w'))
    json.dump(txt2img, open(txt_dir_out + '/txt2img.json', 'w'))
    json.dump(img2txts, open(txt_dir_out + '/img2txts.json', 'w'))


def sample():
    random.seed(seed)

    # load db
    txt_db_in = load_txt_db(txt_dir_in)
    meta = json.load(open(txt_dir_in + '/meta.json'))
    id2len = json.load(open(txt_dir_in + '/id2len.json'))
    txt2img = json.load(open(txt_dir_in + '/txt2img.json'))
    img2txts = json.load(open(txt_dir_in + '/img2txts.json'))

    # sample
    keys = random.sample(list(txt_db_in), sample_num)
    # import ipdb; ipdb.set_trace()
    txt_db_out = {k:txt_db_in[k] for k in keys}
    id2len_out = {k.decode('utf-8'):id2len[k.decode('utf-8')] for k in keys}
    txt2img_out = {k.decode('utf-8'):txt2img[k.decode('utf-8')] for k in keys}
    img2txts_out = {v:[] for v in txt2img.values()}
    for k,v in txt2img_out.items():
        img2txts_out[v].append(k)

    # save db
    save_db(txt_dir_out, txt_db_out, meta, id2len_out, txt2img_out, img2txts_out)




if __name__ == '__main__':
    sample()