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

msgpack_numpy.patch()

sys.path.append('../')
from model.vqa import UniterForVisualQuestionAnswering
from cutmix import load_word_emb_binary, glove_encode, cos_dist
from utils.const import IMG_DIM

NUM_LABELS = 1600
m = torch.nn.Softmax(dim=0)
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

def get_img_item(txt_item, img_txn):
    img_fname = txt_item['img_fname']
    img_dump = img_txn.get(img_fname.encode('utf-8'))
    return msgpack.loads(img_dump, raw=False)

def get_max_sim(txt_item, img_item, labels_emb, glove_dict=None, emb_weight=None, emb_method='UNITER', max_method='Attn'):
    if emb_method == 'UNITER':
        txt_embs = emb_weight[txt_item['input_ids']].cpu().detach().numpy()
    elif emb_method == 'GloVe':
        txt_embs = glove_encode(glove_dict, tokenizer.decode(txt_item['input_ids']))
    lbls = img_item['soft_labels'][:,1:]
    lbl_embs = np.matmul(lbls, labels_emb)
    if 'Attn' in max_method:
        if emb_method == 'UNITER':
            dist = torch.Tensor(np.matmul(txt_embs, lbl_embs.T))
        elif emb_method == 'GloVe':
            dist = cos_dist(txt_embs, lbl_embs)
        # import ipdb; ipdb.set_trace()
        if max_method == 'Attn':
            draw_attn_grid(dist)
            import ipdb; ipdb.set_trace()
            return torch.max(dist)
        elif max_method == 'Attn_Softmax':
            score = m(dist)
            return torch.max(score)

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

    if emb_method == 'UNITER':
        print('Loading UNITER Model')
        model_config = "/data/private/cc/experiment/MMP/UNITER/config/uniter-base.json"
        checkpoint = torch.load("/data/private/cc/experiment/MMP/pretrained_ckpts/pretrained/uniter-base.pt")
        emb_weight = checkpoint['uniter.embeddings.word_embeddings.weight']
    elif emb_method == 'GloVe':
        print('Loading GloVe Model')
        GLOVE_PATH = '/data/share/GloVe'
        glove_dir = '/data/share/GloVe/glove.42B.300d'
        glove_dict = load_word_emb_binary(glove_dir)
        print('GloVe vocab size: %d'%(len(glove_dict)))

    print('Label Embedding ' + emb_method)
    if emb_method == 'UNITER':
        labels_ids = json.load(open('object_labels_ids.json'))
        labels_emb = torch.zeros(NUM_LABELS, emb_weight.shape[1])
        for i in range(NUM_LABELS):
            if len(labels_ids[i]) > 1:
                labels_emb[i] = torch.mean(emb_weight[labels_ids[i]])
            else:
                labels_emb[i] = emb_weight[labels_ids[i][0]]
        labels_emb = labels_emb.cpu().detach().numpy()
    elif emb_method == 'GloVe':
        with open('object_labels.txt') as fin:
            obj_labels = [re.split(',| ', line.strip()) for line in fin]
        labels_emb = torch.zeros(NUM_LABELS, len(glove_dict['the']))
        for i in range(NUM_LABELS):
            if len(obj_labels[i]) > 1:
                label_emb = torch.Tensor([glove_dict[lbl] for lbl in obj_labels[i] if lbl in glove_dict])
                labels_emb[i] = torch.mean(label_emb)
            else:
                labels_emb[i] = torch.Tensor(glove_dict[obj_labels[i][0]])
        labels_emb = labels_emb.cpu().detach().numpy()

    # read from txt db
    txt_env_in = lmdb.open(txt_dir_in)
    txt_txn_in = txt_env_in.begin()
    txt_db = {}

    for key, value in txt_txn_in.cursor():
        txt_db[key] = value
    print('txt db length:', len(txt_db))
    txt_env_in.close()

    img_env_in = lmdb.open(img_dir_in + img_db_name, readonly=True)
    img_txn_in = img_env_in.begin()

    cnt = 0
    for k, v in txt_db.items():
        txt_item = msgpack.loads(decompress(v), raw=False)
        img_item = get_img_item(txt_item, img_txn_in)
        if emb_method == 'UNITER':
            max_sim = get_max_sim(txt_item, img_item, labels_emb, emb_weight=emb_weight, emb_method=emb_method)
        elif emb_method == 'GloVe':
            max_sim = get_max_sim(txt_item, img_item, labels_emb, glove_dict=glove_dict, emb_method=emb_method)
        # import ipdb; ipdb.set_trace()
        # stat_var[txt_item['target']['labels'][0]].append(score_var.item())
        # stat_mean[txt_item['target']['labels'][0]].append(score_mean.item())

        if cnt % 1000 == 0:
            print("Sampled ", cnt)
        cnt += 1

    img_env_in.close()

    # json.dump(stat_var, open('statistic_score_var.json', 'w'))
    # json.dump(stat_mean, open('statistic_score_mean.json', 'w'))

    # plot
    stat_var = json.load(open('statistic_score_var.json'))
    stat_mean = json.load(open('statistic_score_mean.json'))
    for k, v in stat_mean.items():
        y = np.array(v)
        plt.hist(y, bins=400, histtype='step', range=(0, 0.6));
        plt.savefig('score_mean_label' + str(k))
    plt.clf()
    for k, v in stat_var.items():
        y = np.array(v)
        plt.hist(y, bins=400, histtype='step', range=(0, 0.004));
        plt.savefig('score_var_label' + str(k))
    

if __name__ == '__main__':
    main('GloVe')