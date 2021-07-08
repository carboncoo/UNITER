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
from transformers import AutoTokenizer
from lz4.frame import compress, decompress
from os.path import exists, abspath, dirname

msgpack_numpy.patch()

sys.path.append('../')
from model.vqa import UniterForVisualQuestionAnswering
from utils.const import IMG_DIM

NUM_LABELS = 1600
seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

# img_dir = "/data/share/UNITER/nlvr2/img_db/nlvr2_train/feat_th0.2_max100_min10"
# txt_dir = "/data/share/UNITER/nlvr2/txt_db/nlvr2_train.db"

def get_img_item(txt_item, img_txn):
    img_fname = txt_item['img_fname']
    img_dump = img_txn.get(img_fname.encode('utf-8'))
    return msgpack.loads(img_dump, raw=False)

def get_max_idx(txt_item, img_item, emb_weight, labels_emb):
    txt_embs = emb_weight[txt_item['input_ids']].cpu().detach().numpy()
    lbls = img_item['soft_labels'][:,1:]
    lbl_embs = np.matmul(lbls, labels_emb)
    attn = torch.Tensor(np.matmul(txt_embs, lbl_embs.T))
    return (attn==torch.max(attn)).nonzero()


def cutmix(txt1, txt2, img_txn, emb_weight, labels_emb, cnt):
    txt_item1 = msgpack.loads(decompress(txt1), raw=False)
    txt_item2 = msgpack.loads(decompress(txt2), raw=False)
    img_item1 = get_img_item(txt_item1, img_txn)
    img_item2 = get_img_item(txt_item2, img_txn)

    max_idx1 = get_max_idx(txt_item1, img_item1, emb_weight, labels_emb)[0].tolist()
    max_idx2 = get_max_idx(txt_item2, img_item2, emb_weight, labels_emb)[0].tolist()

    img_fname = 'cutmix_%d_'%(cnt) + txt_item1['img_fname']
    mix_txt_item = {'input_ids': txt_item1['input_ids'],
                    'target': txt_item1['target'],
                    'img_fname': img_fname}
    
    mix_txt_item['input_ids'][max_idx1[0]] = txt_item2['input_ids'][max_idx2[0]]
    mix_img_item = {'features': img_item1['features'].copy(),
                    'norm_bb': img_item1['features'].copy(),
                    'conf': img_item1['conf'].copy(),
                    'soft_labels': img_item1['soft_labels'].copy()}
    # import ipdb; ipdb.set_trace()
    mix_img_item['features'][max_idx1[1]] = img_item2['features'][max_idx2[1]]
    mix_img_item['conf'][max_idx1[1]] = img_item2['conf'][max_idx2[1]]
    mix_img_item['soft_labels'][max_idx1[1]] = img_item2['soft_labels'][max_idx2[1]]

    return mix_txt_item, mix_img_item



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
    m = torch.nn.Softmax(dim=0)
    for k, v in txt_db.items():
        txt_item = msgpack.loads(decompress(v), raw=False)
        txt_embs = emb_weight[txt_item['input_ids']].cpu().detach().numpy()

        img_fname = txt_item['img_fname']
        img_dump = img_txn_in.get(img_fname[0].encode('utf-8'))
        img_item = msgpack.loads(img_dump, raw=False)
        lbls = img_item['soft_labels'][:,:1600]
        lbl_embs = np.matmul(lbls, labels_emb)
        attn = torch.Tensor(np.matmul(txt_embs, lbl_embs.T))
        score = m(attn)

        import ipdb; ipdb.set_trace()


    # write to db

def ve():
    # initialize
    img_dir_in = "/data/share/UNITER/ve/img_db/flickr30k"
    txt_dir_in = "/data/share/UNITER/ve/txt_db/ve_train.db"
    img_db_name = "/feat_th0.2_max100_min10"
    # print('Loading Tokenizer')
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    m = torch.nn.Softmax(dim=0)

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
        if len(labels_ids[i]) > 1:
            labels_emb[i] = torch.mean(emb_weight[labels_ids[i]])
        else:
            labels_emb[i] = emb_weight[labels_ids[i][0]]
    labels_emb = labels_emb.cpu().detach().numpy()
    # with open('object_labels_ids.json', 'w') as fout:
    #     json.dump(labels_ids, fout)

    # read from txt db
    txt_env_in = lmdb.open(txt_dir_in)
    txt_txn_in = txt_env_in.begin()
    txt_db = {}
    cnt = 0
    for key, value in txt_txn_in.cursor():
        txt_db[key] = value
        cnt += 1
        # if cnt == 10:
        #     break
    print('txt db length:', len(txt_db))
    txt_env_in.close()

    meta = json.load(open(txt_dir_in + '/meta.json'))
    id2len = json.load(open(txt_dir_in + '/id2len.json'))
    txt2img = json.load(open(txt_dir_in + '/txt2img.json'))
    img2txts = json.load(open(txt_dir_in + '/img2txts.json'))
    nbb_in = json.load(open(img_dir_in + '/nbb_th0.2_max100_min10.json'))

    # img db
    img_env_in = lmdb.open(img_dir_in + img_db_name, readonly=True)
    img_txn_in = img_env_in.begin()

    # debug & test
    # for k, v in txt_db.items():
    #     txt_item = msgpack.loads(decompress(v), raw=False)
    #     txt_embs = emb_weight[txt_item['input_ids']].cpu().detach().numpy()

    #     img_fname = txt_item['img_fname']
    #     img_dump = img_txn_in.get(img_fname.encode('utf-8'))
    #     img_item = msgpack.loads(img_dump, raw=False)
    #     lbls = img_item['soft_labels'][:,1:]
    #     lbl_embs = np.matmul(lbls, labels_emb)
    #     attn = torch.Tensor(np.matmul(txt_embs, lbl_embs.T))
    #     # score = m(attn)
    #     import ipdb; ipdb.set_trace()

        
    # random sample pairs
    seed = 2
    random.seed(seed)
    
    # write to db
    img_dir_out = "/data/share/UNITER/ve/da/seed%d/img_db/flickr30k/feat_th0.2_max100_min10"%(seed)
    txt_dir_out = "/data/share/UNITER/ve/da/seed%d/txt_db/ve_train.db"%(seed)
    if not exists(img_dir_out):
        os.makedirs(img_dir_out)
    else:
        raise ValueError('Found existing DB. Please explicitly remove '
                        'for re-processing')
    if not exists(txt_dir_out):
        os.makedirs(txt_dir_out)
    else:
        raise ValueError('Found existing DB. Please explicitly remove '
                        'for re-processing')

    txt2img_out = {}
    img2txt_out = {}
    nbb_out = {}

    txt_env_out = lmdb.open(txt_dir_out, map_size=int(2e11))
    txt_txn_out = txt_env_out.begin(write=True)
    img_env_out = lmdb.open(img_dir_out, map_size=int(2e11))
    img_txn_out = img_env_out.begin(write=True)

    sample_cnt = 0
    txt_keys = list(txt_db.keys())
    random.shuffle(txt_keys)
    for k, v in txt_db.items():
        # import ipdb; ipdb.set_trace()
        k_sample = txt_keys[sample_cnt]
        while k == k_sample:
            k_sample = random.sample(txt_keys, 1)
        
        mix_txt_item, mix_img_item = cutmix(v, txt_db[k_sample], img_txn_in, emb_weight, labels_emb, sample_cnt)
        mix_txt_key = (str(sample_cnt) + '_').encode('utf-8') + k
        mix_img_key = mix_txt_item['img_fname'].encode('utf-8')
        txt_txn_out.put(mix_txt_key, compress(msgpack.dumps(mix_txt_item, use_bin_type=True)))
        img_txn_out.put(mix_img_key, msgpack.dumps(mix_img_item, use_bin_type=True))
        
        txt2img_out[mix_txt_key] = mix_img_key
        if mix_img_key in img2txt_out:
            img2txt_out[mix_img_key].append(mix_txt_key)
        else:
            img2txt_out[mix_img_key] = [mix_txt_key]
        nbb_out[mix_img_key] = mix_img_item['conf'].shape[0]
        # import ipdb; ipdb.set_trace()

        if sample_cnt % 1000 == 0:
            print("Sampled ", sample_cnt)
        sample_cnt += 1

    print('Mixed %d pairs'%sample_cnt)
    img_env_in.close()
    txt_txn_out.commit()
    txt_env_out.close()
    img_txn_out.commit()
    img_env_out.close()

    json.dump(meta, open(txt_dir_out + '/meta.json', 'w'))
    json.dump(id2len, open(txt_dir_out + '/id2len.json', 'w'))
    json.dump(txt2img_out, open(txt_dir_out + '/txt2img.json', 'w'))
    json.dump(img2txt_out, open(txt_dir_out + '/img2txts.json', 'w'))
    json.dump(nbb_out, open('/data/share/UNITER/ve/da/seed%d/img_db/flickr30k/nbb_th0.2_max100_min10.json', 'w'))
    

if __name__ == "__main__":
	ve()