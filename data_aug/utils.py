import io
import os
import re
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

# cons
IMG_DIM = 2048
NUM_LABELS = 1600

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

def glove_encode(glove_dict, sentence):
    return np.array([glove_dict[w.lower()] if w.lower() in glove_dict else np.ones(len(glove_dict['the']))*1e-6 for w in glove_tokenize(sentence)])

def glove_tokenize(sentence):
    repl = ['.', ',']
    for r in repl:
        sentence = sentence.replace(r, '')
    return re.split("'| ", sentence)

def cos_dist(a, b, eps=1e-6):
    a = torch.Tensor(a)
    b = torch.Tensor(b)
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0,1))

def load_glove_emb_binary(embedding_file_name_w_o_suffix):
    print("Loading binary word embedding from {0}.vocab and {0}.npy".format(embedding_file_name_w_o_suffix))

    with open(embedding_file_name_w_o_suffix + '.vocab', 'r', encoding='utf-8') as f_in:
        index2word = [line.strip() for line in f_in]

    wv = np.load(embedding_file_name_w_o_suffix + '.npy')
    word_embedding_map = {}
    for i, w in enumerate(index2word):
        word_embedding_map[w] = wv[i]

    return word_embedding_map

def load_word_emb(emb_method='UNITER'):
    print('Loading %s Model'%(emb_method))
    if emb_method == 'UNITER':
        model_config = "/data/private/cc/experiment/MMP/UNITER/config/uniter-base.json"
        checkpoint = torch.load("/data/private/cc/experiment/MMP/pretrained_ckpts/pretrained/uniter-base.pt")
        emb_weight = checkpoint['uniter.embeddings.word_embeddings.weight']
    elif emb_method == 'GloVe':
        GLOVE_PATH = '/data/share/GloVe'
        glove_dir = '/data/share/GloVe/glove.42B.300d'
        emb_weight = load_glove_emb_binary(glove_dir)
        print('GloVe vocab size: %d'%(len(emb_weight)))
    return emb_weight

def load_label_emb(emb_weight, emb_method='UNITER'):
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
        labels_emb = torch.zeros(NUM_LABELS, len(emb_weight['the']))
        for i in range(NUM_LABELS):
            if len(obj_labels[i]) > 1:
                label_emb = torch.Tensor([emb_weight[lbl] for lbl in obj_labels[i] if lbl in emb_weight])
                labels_emb[i] = torch.mean(label_emb)
            else:
                labels_emb[i] = torch.Tensor(emb_weight[obj_labels[i][0]])
        labels_emb = labels_emb.cpu().detach().numpy()
    return labels_emb