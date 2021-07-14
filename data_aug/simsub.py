import io
import os
import json
from tqdm import tqdm
from collections import defaultdict
import lmdb
from lz4.frame import compress, decompress
import torch
import numpy as np
import random 
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


def create_class_region_mapping(img_db_txn, keys, n_classes=1601):
    region2class = {}
    class2region = [[] for i in range(1601)]
    for key in tqdm(keys):
        img = load_single_img(img_db_txn, key)
        soft_labels = img['soft_labels']
        max_values = soft_labels.max(axis=-1)
        max_idxs = soft_labels.argmax(axis=-1)
        region2class[key] = []
        for i, idx, value in zip(range(max_idxs.shape[0]), max_idxs, max_values):
            region2class[key].append((idx, value))
            if value < 0.5:
                continue
            new_key = f'{key}${i}'
            class2region[idx].append(new_key)
    
    with open('region2class.mp', 'wb') as f1, open('class2region.mp', 'wb') as f2:
        msgpack.dump(region2class, f1)
        msgpack.dump(class2region, f2)
    return region2class, class2region

from PIL import Image, ImageFont, ImageDraw, ImageEnhance

origin_img_dir = '/data/share/UNITER/origin_imgs/flickr30k/flickr30k-images/'

def draw_bounding_box(img_name, img_bb, outline=(255, 0, 0, 255)):
    source_img = Image.open(origin_img_dir + img_name).convert("RGB")
    width, height = source_img.size

    draw = ImageDraw.Draw(source_img, 'RGBA')
    p1 = (width*img_bb[0], height*img_bb[1])
    p2 = (width*img_bb[2], height*img_bb[3])
    draw.rectangle((p1, p2), outline=outline, width=2)
    return source_img

def split_region_key(key):
    img_name = key.split('_')[-1].split('.')[0].lstrip('0') + '.jpg'
    img_key = key.split('$')[0]
    bb_idx = int(key.split('$')[-1])
    return img_name, img_key, bb_idx

def draw_region(key, img_db_txn):
    img_name, img_key, bb_idx = split_region_key(key)
    bb = load_single_img(img_db_txn, img_key)['norm_bb'][bb_idx]
    img = draw_bounding_box(img_name, bb)
    return img

def substitute_txt(original_txt):
    return original_txt
    
def substitute(img_db_txn, txt_db, class2region, out_dir, seed=42):
    random.seed(seed)
    txt_env_out = lmdb.open(out_dir, map_size=int(1e11))
    txt_txn_out = txt_env_out.begin(write=True)
    sample_cnt = 0
    for k, data in tqdm(txt_db.items()):
        txt_data = load_single_txt(data)
        img_fname = txt_data['img_fname']
        img_data = load_single_img(img_db_txn, img_fname)
        region_soft_labels = img_data['soft_labels']
        max_idx = np.unravel_index(region_soft_labels.argmax(), region_soft_labels.shape) # [row, col]
        candidates = class2region[max_idx[1]]
        sub_region_key = random.sample(candidates, 1)[0]
        sub_region_feat = load_single_region(img_db_txn, sub_region_key)
        
        txt_data['mix_index'] = max_idx[0]
        txt_data['mix_feature'] = sub_region_feat
        txt_data['mix_region'] = sub_region_key
        
        mix_txt_key = str(sample_cnt) + '_' + k.decode('utf-8')
        txt_txn_out.put(mix_txt_key.encode('utf-8'), compress(msgpack.dumps(txt_data, use_bin_type=True)))
        
        if sample_cnt % 1000 == 0:
            txt_txn_out.commit()
            txt_txn_out = txt_env_out.begin(write=True)
        
        sample_cnt += 1
    
    txt_txn_out.commit()
    txt_env_out.close()

def get_concat_h(im1, im2, sentences=None):
    height = max(im1.height, im2.height)
    if sentences is not None:
        height += 50 * len(sentences)
    dst = Image.new('RGB', (im1.width + im2.width, height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    
    fnt = ImageFont.truetype(font='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', size=20)
    if sentences is not None:
        draw = ImageDraw.Draw(dst)
        left, top = 0, max(im1.height, im2.height)
        for sent in sentences:
            draw.text((left, top), sent, (255, 255, 255), font=fnt)
            top += 50
    return dst

def sample(img_db_txn, mixed_txt_db, sample_n=10, out_dir='./sample'):
    keys = mixed_txt_db.keys()
    sampled_keys = random.sample(keys, sample_n)
    for key in sampled_keys:
        txt_data = load_single_txt(mixed_txt_db[key])
        sentence = ' '.join(txt_data['toked_hypothesis'])
        
        original_region_key = txt_data['img_fname'] + f"${txt_data['mix_index']}"
        sub_region_key = txt_data['mix_region']
        original_region = draw_region(original_region_key, img_db_txn)
        sub_region = draw_region(sub_region_key, img_db_txn)
        
        output_img = get_concat_h(original_region, sub_region, [sentence])
        output_img.save(os.path.join(out_dir, f'{key}.jpeg'), 'JPEG')

if __name__ == '__main__':
    # name2nbb, img_db_txn = load_img_db('/data/share/UNITER/ve/img_db/flickr30k')
    # txt_db = load_txt_db('/data/share/UNITER/ve/txt_db/ve_train.db')
    
    # CREATE
    # region2class, class2region = create_class_region_mapping(img_db_txn, list(name2nbb.keys())) # it takes about 35 seconds
    
    # LOAD
    # with open('region2class.mp', 'rb') as f1, open('class2region.mp', 'rb') as f2:
    #     region2class = msgpack.load(f1)
    #     class2region = msgpack.load(f2)

    # SUBSTITUTE
    # seed = 42
    # out_dir = f'/data/share/UNITER/ve/da/simsub-seed{seed}-1/txt_db/ve_train.db'
    # substitute(img_db_txn, txt_db, class2region, out_dir, seed=seed)
    
    # SAMPLE
    name2nbb, img_db_txn = load_img_db('/data/share/UNITER/ve/img_db/flickr30k')
    seed = 42
    out_dir = f'/data/share/UNITER/ve/da/simsub-seed{seed}-1/txt_db/ve_train.db'
    txt_db = load_txt_db(out_dir)
    sample(img_db_txn, txt_db)
    # import ipdb; ipdb.set_trace()