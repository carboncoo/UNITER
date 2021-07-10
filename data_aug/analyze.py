import os
import sys
import lmdb
import json
import torch
import pickle
import random
import msgpack
import numpy as np
import msgpack_numpy
# from transformers import AutoTokenizer
from lz4.frame import compress, decompress
from os.path import exists, abspath, dirname
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

import pprint
pp = pprint.PrettyPrinter()

msgpack_numpy.patch()
origin_img_dir = '/data/share/UNITER/origin_imgs/flickr30k/flickr30k-images/'

def load_txt_db(db_dir):
    # db loading
    env_in = lmdb.open(db_dir, readonly=True, create=False)
    txn_in = env_in.begin()
    db = {}
    for key, value in txn_in.cursor():
        db[key] = value
    print('db length:', len(db))  # db length: 443757
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


def load_single_img(txn, file_name, compress=False):
    # load single image with its file_name
    dump = txn.get(file_name.encode('utf-8'))
    if compress:
        with io.BytesIO(dump) as reader:
            img_dump = np.load(reader, allow_pickle=True)
            img_dump = {'features': img_dump['features'],
                        'norm_bb': img_dump['norm_bb']}
    else:
        img_dump = msgpack.loads(dump, raw=False)
    return img_dump

def draw_bounding_box(img_name, img_bb):
    source_img = Image.open(origin_img_dir + img_name).convert("RGB")
    width, height = source_img.size

    draw = ImageDraw.Draw(source_img, 'RGBA')
    p1 = (width*img_bb[0], height*img_bb[1])
    p2 = (width*img_bb[2], height*img_bb[3])
    # draw.rectangle((p1, p2), fill=(200, 100, 0, 127))
    draw.rectangle((p1, p2), outline=(0, 0, 0, 127), width=2)
    # draw.rectangle((, ), fill="black")
    # draw.text((img_bb[0], img_bb[1]), "something123", font=ImageFont.truetype("font_path123"))

    source_img.save('bb_' + img_name, "JPEG")

def main():
    NUM_LABELS = 1600
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    id2tok = json.load(open('id2tok.json'))
    labels_ids = json.load(open('object_labels_ids.json'))

    def convert_ids_to_tokens(i):
        if isinstance(i, int):
            return id2tok[str(i)]
        else:
            i = list(i)
        return [id2tok[str(ii)] for ii in i]

    def get_label_str(i):
        if isinstance(i, int):
            return convert_ids_to_tokens(labels_ids[i])
        else:
            i = list(i)
        return [convert_ids_to_tokens(labels_ids[ii]) if ii > 0 else '[BACKGROUND]' for ii in i]

    def get_hard_labels(soft_labels, top_k=3):
        if len(soft_labels.shape) < 2:
            soft_labels = soft_labels.reshape(1, -1)
        sorted_labels = soft_labels.argsort(axis=-1)[:, ::-1][:, :top_k]
        sorted_labels = sorted_labels - 1
        res = []
        for l in sorted_labels:
            res.append(get_label_str(l))
        return res

    checkpoint = torch.load(
        "/data/private/cc/experiment/MMP/pretrained_ckpts/pretrained/uniter-base.pt")
    emb_weight = checkpoint['uniter.embeddings.word_embeddings.weight']

    txt_db_old = load_txt_db('/data/share/UNITER/ve/txt_db/ve_train.db')
    txt_db_new = load_txt_db(
        '/data/share/UNITER/ve/da/seed3/txt_db/ve_train.db')
    name2nbb, img_db_txn = load_img_db('/data/share/UNITER/ve/img_db/flickr30k')

    def display(k):
        d1 = msgpack.loads(decompress(txt_db_old[k.split(b'_')[1]]), raw=False)
        d2 = msgpack.loads(decompress(txt_db_new[k]), raw=False)
        # input_1 = tokenizer.convert_ids_to_tokens(d1['input_ids'])
        # input_2 = tokenizer.convert_ids_to_tokens(d2['input_ids'])
        input_1 = convert_ids_to_tokens(d1['input_ids'])
        input_2 = convert_ids_to_tokens(d2['input_ids'])

        hard_labels = get_hard_labels(d2['mix_soft_labels'])

        # img1 = load_single_img(img_db_txn, d1['img_fname'])
        img2 = load_single_img(img_db_txn, d2['img_fname'])
        origin_img_name = str(k).split('_')[1].split('#')[0]
        draw_bounding_box(origin_img_name, img2['norm_bb'][d2['mix_index']])
        # print(img2['norm_bb'][d2['mix_index']])

        return input_1, input_2, hard_labels

    # print(list(txt_db_new.keys())[:10])
    pp.pprint(display(list(txt_db_new.keys())[0]))
    pp.pprint(display(list(txt_db_new.keys())[1]))
    # pp.pprint(display(list(txt_db_new.keys())[2]))



    # import ipdb
    # ipdb.set_trace()


if __name__ == '__main__':
    main()
