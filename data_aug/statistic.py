import os
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
from lz4.frame import compress, decompress
from os.path import exists, abspath, dirname

msgpack_numpy.patch()

sys.path.append('../')
from model.vqa import UniterForVisualQuestionAnswering
from utils.const import IMG_DIM

NUM_LABELS = 1600
m = torch.nn.Softmax(dim=0)

def get_img_item(txt_item, img_txn):
    img_fname = txt_item['img_fname']
    img_dump = img_txn.get(img_fname.encode('utf-8'))
    return msgpack.loads(img_dump, raw=False)

def get_max_attn(txt_item, img_item, emb_weight, labels_emb):
    txt_embs = emb_weight[txt_item['input_ids']].cpu().detach().numpy()
    lbls = img_item['soft_labels'][:,1:]
    lbl_embs = np.matmul(lbls, labels_emb)
    attn = torch.Tensor(np.matmul(txt_embs, lbl_embs.T))
    return torch.max(attn)

def get_score_stat(txt_item, img_item, emb_weight, labels_emb):
    txt_embs = emb_weight[txt_item['input_ids']].cpu().detach().numpy()
    lbls = img_item['soft_labels'][:,1:]
    lbl_embs = np.matmul(lbls, labels_emb)
    attn = torch.Tensor(np.matmul(txt_embs, lbl_embs.T))
    score = m(attn)
    return torch.mean(score), torch.var(score)

def main():
    # initialize
    # img_dir_in = "/data/share/UNITER/ve/img_db/flickr30k"
    # txt_dir_in = "/data/share/UNITER/ve/txt_db/ve_train.db"
    # img_db_name = "/feat_th0.2_max100_min10"
    # stat_mean = {0: [], 1:[], 2:[]}
    # stat_var = {0: [], 1:[], 2:[]}

    # print('Loading UNITER Model')
    # model_config = "/data/private/cc/experiment/MMP/UNITER/config/uniter-base.json"
    # checkpoint = torch.load("/data/private/cc/experiment/MMP/pretrained_ckpts/pretrained/uniter-base.pt")
    # model = UniterForVisualQuestionAnswering.from_pretrained(
    #         model_config, state_dict=checkpoint,
    #         img_dim=IMG_DIM, num_answer= 3129,
    #         da_type=None)
    # emb_weight = model.uniter.embeddings.word_embeddings.weight

    # print('Label Embedding')
    # labels_ids = json.load(open('object_labels_ids.json'))
    # labels_emb = torch.zeros(NUM_LABELS, emb_weight.shape[1])
    # for i in range(NUM_LABELS):
    #     if len(labels_ids[i]) > 1:
    #         labels_emb[i] = torch.mean(emb_weight[labels_ids[i]])
    #     else:
    #         labels_emb[i] = emb_weight[labels_ids[i][0]]
    # labels_emb = labels_emb.cpu().detach().numpy()

    # # read from txt db
    # txt_env_in = lmdb.open(txt_dir_in)
    # txt_txn_in = txt_env_in.begin()
    # txt_db = {}

    # for key, value in txt_txn_in.cursor():
    #     txt_db[key] = value
    # print('txt db length:', len(txt_db))
    # txt_env_in.close()

    # img_env_in = lmdb.open(img_dir_in + img_db_name, readonly=True)
    # img_txn_in = img_env_in.begin()

    # cnt = 0
    # for k, v in txt_db.items():
    #     txt_item = msgpack.loads(decompress(v), raw=False)
    #     img_item = get_img_item(txt_item, img_txn_in)
    #     score_mean, score_var = get_score_stat(txt_item, img_item, emb_weight, labels_emb)
    #     # import ipdb; ipdb.set_trace()
    #     stat_var[txt_item['target']['labels'][0]].append(score_var.item())
    #     stat_mean[txt_item['target']['labels'][0]].append(score_mean.item())

    #     if cnt % 1000 == 0:
    #         print("Sampled ", cnt)
    #     cnt += 1

    # img_env_in.close()

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
    main()