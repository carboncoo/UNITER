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
from sklearn.cluster import k_means
from transformers import AutoTokenizer
from lz4.frame import compress, decompress
from os.path import exists, abspath, dirname

msgpack_numpy.patch()

sys.path.append('../')
from model.vqa import UniterForVisualQuestionAnswering

IMG_DIM = 2048
NUM_LABELS = 1600
seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

m = torch.nn.Softmax(dim=0)
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

def convert_to_binary(embedding_path):
    f = open(embedding_path + ".txt", 'r', encoding='utf-8')
    wv = []

    with open(embedding_path + ".vocab", "w", encoding='utf-8') as vocab_write:
        count = 0
        for line in f:
            splitlines = line.split()
            vocab_write.write(splitlines[0].strip())
            vocab_write.write("\n")
            wv.append([float(val) for val in splitlines[1:]])
        count += 1

    np.save(embedding_path + ".npy", np.array(wv))

def load_word_emb_binary(embedding_file_name_w_o_suffix):
    print("Loading binary word embedding from {0}.vocab and {0}.npy".format(embedding_file_name_w_o_suffix))

    with open(embedding_file_name_w_o_suffix + '.vocab', 'r', encoding='utf-8') as f_in:
        index2word = [line.strip() for line in f_in]

    wv = np.load(embedding_file_name_w_o_suffix + '.npy')
    word_embedding_map = {}
    for i, w in enumerate(index2word):
        word_embedding_map[w] = wv[i]

    return word_embedding_map

def glove_tokenize(sentence):
    repl = ['.', ',']
    for r in repl:
        sentence = sentence.replace(r, '')
    return re.split("'| ", sentence)

def glove_encode(glove_dict, sentence):
    return np.array([glove_dict[w.lower()] if w.lower() in glove_dict else np.ones(len(glove_dict['the']))*1e-6 for w in glove_tokenize(sentence)])

def cos_dist(a, b, eps=1e-6):
    a = torch.Tensor(a)
    b = torch.Tensor(b)
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0,1))

def get_img_item(txt_item, img_txn):
    img_fname = txt_item['img_fname']
    img_dump = img_txn.get(img_fname.encode('utf-8'))
    return msgpack.loads(img_dump, raw=False)

def get_max_idx(txt_item, img_item, labels_emb, glove_dict=None, emb_weight=None, emb_method='UNITER', max_method='Attn'):
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
            return (dist==torch.max(dist)).nonzero()
        elif max_method == 'Attn_Softmax':
            score = m(dist)
            return (score==torch.max(score)).nonzero()

    # import ipdb; ipdb.set_trace()

def cutmix(txt1, txt2, img_txn, labels_emb, glove_dict=None, emb_weight=None, emb_method='UNITER', max_method='Attn'):
    txt_item1 = msgpack.loads(decompress(txt1), raw=False)
    txt_item2 = msgpack.loads(decompress(txt2), raw=False)
    img_item1 = get_img_item(txt_item1, img_txn)
    img_item2 = get_img_item(txt_item2, img_txn)

    max_idx1 = get_max_idx(txt_item1, img_item1, labels_emb, glove_dict=glove_dict, emb_method=emb_method, max_method=max_method)[0].tolist()
    max_idx2 = get_max_idx(txt_item2, img_item2, labels_emb, glove_dict=glove_dict, emb_method=emb_method, max_method=max_method)[0].tolist()

    mix_txt_item = {'input_ids': txt_item1['input_ids'],
                    'target': txt_item1['target'],
                    'img_fname': txt_item1['img_fname'],
                    'mix_img_flk_id': txt_item2['Flikr30kID'],
                    'mix_input_ids': txt_item2['input_ids'],
                    'mix_bb': img_item2['norm_bb'][max_idx2[1]],
                    'mix_index': max_idx1[1],
                    'mix_feature': img_item2['features'][max_idx2[1]],
                    'mix_conf': img_item2['conf'][max_idx2[1]],
                    'mix_soft_labels': img_item2['soft_labels'][max_idx2[1]]}
    if emb_method == 'UNITER':
        mix_txt_item['input_ids'][max_idx1[0]] = txt_item2['input_ids'][max_idx2[0]]
    if emb_method == 'GloVe':
        mix_s = glove_tokenize(tokenizer.decode(txt_item1['input_ids']))
        s2 = glove_tokenize(tokenizer.decode(txt_item2['input_ids']))
        mix_s[max_idx1[0]] = s2[max_idx2[0]]
        mix_txt_item['input_ids'] = tokenizer.encode(' '.join(mix_s))[1:-1]
    # import ipdb; ipdb.set_trace()
    return mix_txt_item

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

def ve(emb_method='UNITER'):
    # initialize
    img_dir_in = "/data/share/UNITER/ve/img_db/flickr30k"
    txt_dir_in = "/data/share/UNITER/ve/da/sample/seed2/txt_db/ve_train.db"
    img_db_name = "/feat_th0.2_max100_min10"
    # print('Loading Tokenizer')
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

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
    # with open('object_labels.txt') as fin:
    #     obj_labels = [line.strip() for line in fin]
    # labels_ids = [tokenizer.encode(label)[1:-1] for label in obj_labels]
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
        
    # random sample pairs
    seed = 2
    random.seed(seed)
    
    # write to db
    txt_dir_out = "/data/share/UNITER/ve/da/sample/seed2/%s/txt_db/ve_train.db"%(emb_method)
    if not exists(txt_dir_out):
        os.makedirs(txt_dir_out)
    else:
        raise ValueError('Found existing DB. Please explicitly remove '
                        'for re-processing')

    id2len_out = {}
    txt2img_out = {}
    img2txt_out = {}

    txt_env_out = lmdb.open(txt_dir_out, map_size=int(1e11))
    txt_txn_out = txt_env_out.begin(write=True)

    sample_cnt = 0
    txt_keys = list(txt_db.keys())
    random.shuffle(txt_keys)
    for k, v in txt_db.items():
        # import ipdb; ipdb.set_trace()
        k_sample = txt_keys[sample_cnt]
        while k == k_sample:
            print('whoooooooops')
            k_sample = random.choice(txt_keys)
        
        if emb_method == 'UNITER':
            mix_txt_item = cutmix(v, txt_db[k_sample], img_txn_in, labels_emb, emb_weight=emb_weight, emb_method=emb_method, max_method='Attn_Softmax')
        elif emb_method == 'GloVe':
            mix_txt_item = cutmix(v, txt_db[k_sample], img_txn_in, labels_emb, glove_dict=glove_dict, emb_method=emb_method, max_method='Attn_Softmax')
        mix_txt_key = str(sample_cnt) + '_' + k.decode('utf-8')
        mix_img_key = mix_txt_item['img_fname']
        txt_txn_out.put(mix_txt_key.encode('utf-8'), compress(msgpack.dumps(mix_txt_item, use_bin_type=True)))
        
        txt2img_out[mix_txt_key] = mix_img_key
        if mix_img_key in img2txt_out:
            img2txt_out[mix_img_key].append(mix_txt_key)
        else:
            img2txt_out[mix_img_key] = [mix_txt_key]
        id2len_out[mix_txt_key] = id2len[k.decode('utf-8')]
        # import ipdb; ipdb.set_trace()

        if sample_cnt % 1000 == 0:
            print("Sampled ", sample_cnt)
            txt_txn_out.commit()
            txt_txn_out = txt_env_out.begin(write=True)
        sample_cnt += 1

    print('Mixed %d pairs'%sample_cnt)
    img_env_in.close()
    txt_txn_out.commit()
    txt_env_out.close()

    json.dump(meta, open(txt_dir_out + '/meta.json', 'w'))
    json.dump(id2len_out, open(txt_dir_out + '/id2len.json', 'w'))
    json.dump(txt2img_out, open(txt_dir_out + '/txt2img.json', 'w'))
    json.dump(img2txt_out, open(txt_dir_out + '/img2txts.json', 'w'))
    

if __name__ == "__main__":
	ve('GloVe')