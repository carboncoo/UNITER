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
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from lz4.frame import compress, decompress
from os.path import exists, abspath, dirname
from analyze import draw_bounding_box
from cutmix import check_pos
from utils import load_word_emb, load_label_emb, load_txt_db, glove_encode, cos_dist, IMG_DIM, NUM_LABELS
msgpack_numpy.patch()


m = torch.nn.Softmax(dim=0)
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

def get_img_item(txt_item, img_txn):
    img_fname = txt_item['img_fname']
    img_dump = img_txn.get(img_fname.encode('utf-8'))
    return msgpack.loads(img_dump, raw=False)

def get_max_sim(txt_item, img_item, labels_emb, emb_weight=None, emb_method='UNITER', max_method='Attn'):
    if emb_method == 'UNITER':
        txt_embs = emb_weight[txt_item['input_ids']].cpu().detach().numpy()
    elif emb_method == 'GloVe':
        txt_embs = glove_encode(emb_weight, tokenizer.decode(txt_item['input_ids']))
    lbls = img_item['soft_labels'][:,1:]
    lbl_embs = np.matmul(lbls, labels_emb)
    if 'Attn' in max_method:
        if emb_method == 'UNITER':
            dist = torch.Tensor(np.matmul(txt_embs, lbl_embs.T))
        elif emb_method == 'GloVe':
            dist = cos_dist(txt_embs, lbl_embs)
        if max_method == 'Attn':
            # draw_attn_grid(dist)
            return torch.max(dist)
        elif max_method == 'Attn_Softmax':
            score = m(dist)
            return torch.max(score)

def get_max_idx(txt_item, img_item, labels_emb, threshold=0.8, pos=None, emb_weight=None, emb_method='UNITER', max_method='Attn'):
    if emb_method == 'UNITER':
        txt_embs = emb_weight[txt_item['input_ids']].cpu().detach().numpy()
    elif emb_method == 'GloVe':
        txt_embs = glove_encode(emb_weight, tokenizer.decode(txt_item['input_ids']))
    lbls = img_item['soft_labels'][:,1:]
    lbl_embs = np.matmul(lbls, labels_emb)
    if 'Attn' in max_method:
        if emb_method == 'UNITER':
            dist = torch.Tensor(np.matmul(txt_embs, lbl_embs.T))
        elif emb_method == 'GloVe':
            dist = cos_dist(txt_embs, lbl_embs)
        if max_method == 'Attn':
            max_dist = torch.max(dist)
            max_idx = (dist==torch.max(dist)).nonzero()[0].tolist()
            if (max_dist > threshold).item():
                if pos != None:
                    if not check_pos(txt_item, max_idx, pos):
                        im = draw_bounding_box(txt_item['Flikr30kID'], img_item['norm_bb'][max_idx[1]], (200, 100, 0, 255))
                        im.save('pos_' + txt_item['Flikr30kID'], 'JPEG')
                        import ipdb; ipdb.set_trace()
                        return None
                return max_idx
            else:
                # im = draw_bounding_box(txt_item['Flikr30kID'], img_item['norm_bb'][max_idx[1]])
                # im.save('threshold_' + txt_item['Flikr30kID'], 'JPEG')
                # import ipdb; ipdb.set_trace()
                return None
        elif max_method == 'Attn_Softmax':
            score = m(dist)
            return (score==torch.max(score)).nonzero()[0].tolist()

def get_score_stat(txt_item, img_item, emb_weight, labels_emb):
    txt_embs = emb_weight[txt_item['input_ids']].cpu().detach().numpy()
    lbls = img_item['soft_labels'][:,1:]
    lbl_embs = np.matmul(lbls, labels_emb)
    attn = torch.Tensor(np.matmul(txt_embs, lbl_embs.T))
    score = m(attn)
    return torch.mean(score), torch.var(score)

def draw_attn_grid(attn):
    plt.imshow(attn, cmap=plt.get_cmap('Greens'), alpha=0.5)
    plt.savefig('Attn.png')

def main(emb_method='UNITER'):
    # initialize
    img_dir_in = "/data/share/UNITER/ve/img_db/flickr30k"
    txt_dir_in = "/data/share/UNITER/ve/txt_db/ve_train.db"
    img_db_name = "/feat_th0.2_max100_min10"
    stat_mean = {0: [], 1:[], 2:[]}
    stat_var = {0: [], 1:[], 2:[]}
    stat_max_sim = []

    # load embs
    emb_weight = load_word_emb(emb_method)
    labels_emb = load_label_emb(emb_weight, emb_method)

    # read from txt db
    txt_db = load_txt_db(txt_dir_in)

    img_env_in = lmdb.open(img_dir_in + img_db_name, readonly=True)
    img_txn_in = img_env_in.begin()

    cnt = 0
    for k, v in txt_db.items():
        txt_item = msgpack.loads(decompress(v), raw=False)
        img_item = get_img_item(txt_item, img_txn_in)
        max_idx = get_max_idx(txt_item, img_item, labels_emb, threshold=0.8, pos='NN', emb_weight=emb_weight, emb_method=emb_method, max_method='Attn')
        # max_sim = get_max_sim(txt_item, img_item, labels_emb, emb_weight=emb_weight, emb_method=emb_method)
        # import ipdb; ipdb.set_trace()
        # stat_var[txt_item['target']['labels'][0]].append(score_var.item())
        # stat_mean[txt_item['target']['labels'][0]].append(score_mean.item())
        # stat_max_sim.append(max_sim.item())
        if cnt % 1000 == 0:
            print("Sampled ", cnt)
        cnt += 1

    img_env_in.close()

    # json.dump(stat_max_sim, open('statistic_max_sim.json', 'w'))
    # json.dump(stat_var, open('statistic_score_var.json', 'w'))
    # json.dump(stat_mean, open('statistic_score_mean.json', 'w'))

    # plot
    # stat_var = json.load(open('statistic_score_var.json'))
    # stat_mean = json.load(open('statistic_score_mean.json'))
    # for k, v in stat_mean.items():
    #     y = np.array(v)
    #     plt.hist(y, bins=400, histtype='step', range=(0, 0.6));
    #     plt.savefig('score_mean_label' + str(k))
    # plt.clf()
    # for k, v in stat_var.items():
    #     y = np.array(v)
    #     plt.hist(y, bins=400, histtype='step', range=(0, 0.004));
    #     plt.savefig('score_var_label' + str(k))

    # plot max sim
    # y = np.array(stat_max_sim)
    # plt.hist(y, bins=400, histtype='step')
    # plt.savefig('max_sim')
    

if __name__ == '__main__':
    main('GloVe')