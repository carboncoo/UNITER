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
from transformers import AutoTokenizer
from lz4.frame import compress, decompress
from os.path import exists, abspath, dirname

msgpack_numpy.patch()

sys.path.append('../')
from model.vqa import UniterForVisualQuestionAnswering
from utils.const import IMG_DIM

NUM_LABELS = 1600
seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

img_dir = "/data/share/UNITER/nlvr2/img_db/nlvr2_train/feat_th0.2_max100_min10"
txt_dir = "/data/share/UNITER/nlvr2/txt_db/nlvr2_train.db"

def nlvr2():
    # initialize
    # print('Loading Tokenizer')
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    print('Loading UNITER Model')
    model_config = "/data/private/cc/experiment/MMP/UNITER/config/uniter-base.json"
    checkpoint = torch.load("/data/private/cc/experiment/MMP/pretrained_ckpts/pretrained/uniter-base.pt")
    model = UniterForVisualQuestionAnswering.from_pretrained(
            model_config, state_dict=checkpoint,
            img_dim=IMG_DIM, num_answer= 3129,
            da_type=None)
    emb_weight = model.uniter.embeddings.word_embeddings.weight

    print('Label Embedding')
    # with open('object_labels.txt') as fin:
    #     obj_labels = [line.strip() for line in fin]
    # labels_ids = [tokenizer.encode(label)[1:-1] for label in obj_labels]
    labels_ids = json.load(open('object_labels_ids.json'))
    labels_emb = torch.zeros(NUM_LABELS, emb_weight.shape[1])
    for i in range(NUM_LABELS):
        labels_emb[i] = torch.mean(emb_weight[labels_ids[i]])
    labels_emb = labels_emb.cpu().detach().numpy()
    # with open('object_labels_ids.json', 'w') as fout:
    #     json.dump(labels_ids, fout)

    # read from txt db
    txt_env_in = lmdb.open(txt_dir)
    txt_txn_in = txt_env_in.begin()
    txt_db = {}
    cnt = 0
    for key, value in txt_txn_in.cursor():
        txt_db[key] = value
        cnt += 1
        if cnt == 10:
            break
    print('txt db length:', len(txt_db))
    txt_env_in.close()

    meta = json.load(open(txt_dir + '/meta.json'))
    id2len = json.load(open(txt_dir + '/id2len.json'))
    txt2img = json.load(open(txt_dir + '/txt2img.json'))

    # img db
    img_env_in = lmdb.open(img_dir, readonly=True)
    img_txn_in = img_env_in.begin()

    # debug & test
    for k, v in txt_db.items():
        txt_item = msgpack.loads(decompress(v), raw=False)
        txt_embs = [emb_weight[idx].cpu().detach().numpy() for idx in txt_item['input_ids']]

        img_fname = txt_item['img_fname']
        img_dump = img_txn_in.get(img_fname[0].encode('utf-8'))
        img_item = msgpack.loads(img_dump, raw=False)
        labels = img_item['soft_labels']
        for l in labels:
            l_weight = np.expand_dims(l[:1600], axis=0)
            l_emb = np.squeeze(np.matmul(l_weight, labels_emb))
            dists = [np.linalg.norm(l_emb - t_emb) for t_emb in txt_embs]
            import ipdb; ipdb.set_trace()


    # write to db

if __name__ == "__main__":
	nlvr2()