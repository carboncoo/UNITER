import io
import os
import re
import json
from tqdm import tqdm
from collections import defaultdict
import lmdb
from lz4.frame import compress, decompress
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np 
import msgpack
import msgpack_numpy
msgpack_numpy.patch()


# cons
IMG_DIM = 2048
NUM_LABELS = 1600

meta = {
    'CLS': 101,
    'SEP': 102,
    'MASK': 103
}

import glob
import shutil
import random

from PIL import Image, ImageFont, ImageDraw, ImageEnhance

class TxtLmdb(object):
    def __init__(self, db_dir, readonly=True):
        self.readonly = readonly
        if readonly:
            # training
            self.env = lmdb.open(db_dir,
                                 readonly=True, create=False)
            self.txn = self.env.begin(buffers=True)
            self.write_cnt = None
        else:
            # prepro
            self.env = lmdb.open(db_dir, readonly=False, create=True,
                                 map_size=4 * 1024**4)
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0
        
        self.infos = {
            'meta': json.load(open(os.path.join(db_dir, 'meta.json'))),
            'id2len': json.load(open(os.path.join(db_dir, 'id2len.json'))),
            'txt2img': json.load(open(os.path.join(db_dir, 'txt2img.json'))),
            'img2txts': json.load(open(os.path.join(db_dir, 'img2txts.json')))
        }
            
        self._keys = None
        self.db = None

    def __del__(self):
        if self.write_cnt:
            self.txn.commit()
        self.env.close()

    def __getitem__(self, key):
        if isinstance(key, int):
            if self._keys is None:
                _keys = []
                for k, v in self.txn.cursor():
                    _keys.append(k.tobytes())
                self._keys = _keys
            return self.__getitem__(self._keys[key])
        else:
            if isinstance(key, str):
                key = key.encode('utf-8')
            return msgpack.loads(decompress(self.txn.get(key)),
                                raw=False)

    def __setitem__(self, key, value):
        # NOTE: not thread safe
        if self.readonly:
            raise ValueError('readonly text DB')
        ret = self.txn.put(key.encode('utf-8'),
                           compress(msgpack.dumps(value, use_bin_type=True)))
        self.write_cnt += 1
        if self.write_cnt % 1000 == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0
        return ret
    
    def load_dict(self):
        if self.db:
            return self.db
        self.db = {}
        for key, value in self.txn.cursor():
            self.db[key.tobytes()] = value
        return self.db
    
    @staticmethod
    def save_db(db_dir, db, infos=None):
        print(f'Saving txt_db: {db_dir} ({len(db)})')
        os.makedirs(db_dir, exist_ok=True)
        env = lmdb.open(db_dir, readonly=False, create=True,
                                map_size=4 * 1024**4)
        txn = env.begin(write=True)
        
        write_cnt = 0
        
        keys = sorted(list(db.keys()))
        
        for k in keys:
            v = db[k]
            if isinstance(k, str):
                k = k.encode('utf-8')
            txn.put(k, compress(msgpack.dumps(v, use_bin_type=True)))
            write_cnt += 1
            if write_cnt % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
            if write_cnt % 10000 == 0:
                print(f'{write_cnt}...')
        
        txn.commit()
        env.close()
          
        if infos is not None:
            for k, v in infos.items():
                json.dump(v, open(os.path.join(db_dir, f'{k}.json'), 'w'))
        # if meta is not None:
        #     json.dump(meta, open(os.path.join(db_dir, 'meta.json', 'w')))
        # if id2len is not None:
        #     json.dump(id2len, open(os.path.join(db_dir, 'id2len.json', 'w')))
        # if txt2img is not None:
        #     json.dump(txt2img, open(os.path.join(db_dir, 'txt2img.json', 'w')))
        # if img2txts is not None:
        #     json.dump(img2txts, open(os.path.join(db_dir, 'img2txts.json', 'w')))   
        
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

def get_vqa_target(txt_item, num_answers=3):
    target = torch.zeros(num_answers)
    labels = txt_item['target']['labels']
    scores = txt_item['target']['scores']
    try:
        if labels and scores:
            target.scatter_(0, torch.tensor(labels), torch.tensor(scores))
    except:
        pass
    return target

def combine_inputs(*inputs):
    input_ids = [meta['CLS']]
    for ids in inputs:
        input_ids.extend(ids + [meta['SEP']])
    return torch.tensor(input_ids).unsqueeze(0)

def get_gather_index(txt_lens, num_bbs, batch_size, max_len, out_size):
    assert len(txt_lens) == len(num_bbs) == batch_size
    gather_index = torch.arange(0, out_size, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

    for i, (tl, nbb) in enumerate(zip(txt_lens, num_bbs)):
        gather_index.data[i, tl:tl+nbb] = torch.arange(max_len, max_len+nbb,
                                                       dtype=torch.long).data
    return gather_index

def load_as_batch(txt_item, img_item):
    # img input
    img_feat = torch.Tensor(img_item['features']).unsqueeze(0)
    bb = torch.Tensor(img_item['norm_bb'])
    img_pos_feat = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)

    if 'mix_index' in txt_item:
        mix_index = txt_item['mix_index']
        img_feat.data[0, mix_index] = torch.from_numpy(txt_item['mix_feature'])

    # txt input
    input_ids = txt_item['input_ids']
    input_ids = combine_inputs(input_ids)

    targets = get_vqa_target(txt_item)
    attn_masks = torch.ones(len(input_ids[0]) + img_feat.size(1), dtype=torch.long).unsqueeze(0)

    txt_lens = [i.size(0) for i in input_ids]
    # input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    # targets = torch.stack(target, dim=0)

    num_bbs = [f.size(0) for f in img_feat]
    # img_feat = pad_tensors(img_feat, num_bbs)
    # img_pos_feat = pad_tensors(img_pos_feat, num_bbs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets}
    return batch

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

##################### Visualization #####################

def split_region_key(key):
    img_name = key.split('_')[-1].split('.')[0].lstrip('0') + '.jpg'
    img_key = key.split('$')[0]
    if '$' in key:
        bb_idx = int(key.split('$')[-1])
    else:
        bb_idx = None
    return img_name, img_key, bb_idx

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

def fname2imgid(fname):
    return re.match(r'.*_0*([0-9]+).npz', fname).group(1)

def imgid2fname(imgid):
    return f'flickr30k_{imgid.zfill(12)}.npz'

class VizOutput:
    
    origin_img_dir = '/data/share/UNITER/origin_imgs/flickr30k/flickr30k-images/'
    
    def __init__(self, img_name):
        self.img = Image.open(self.origin_img_dir + img_name).convert("RGB")
        self.width, self.height = self.img.size
        self.draw = ImageDraw.Draw(self.img, 'RGBA')
        
    def draw_bounding_box(self, img_bb, outline=(255, 0, 0, 255)):
        p1 = (self.width*img_bb[0], self.height*img_bb[1])
        p2 = (self.width*img_bb[2], self.height*img_bb[3])
        self.draw.rectangle((p1, p2), outline=outline, width=2)
    
    def save(self, path):
        self.img.save(path, 'JPEG')
  
    @classmethod
    def get_img_output(cls, key, img_db_txn):
        ''' [img_idx].jpg
        '''
        img_name, img_key, bb_idx = split_region_key(key)
        bbs = load_single_img(img_db_txn, img_key)['norm_bb']
        img = cls(img_name)
        for bb in bbs:
            img.draw_bounding_box(bb)
        return img
    
    @classmethod
    def get_region_output(cls, key, img_db_txn):
        ''' [img_idx].jpg$[region_idx]
        '''
        img_name, img_key, bb_idx = split_region_key(key)
        bb = load_single_img(img_db_txn, img_key)['norm_bb'][bb_idx]
        img = cls(img_name)
        img.draw_bounding_box(bb)
        return img
    
    @staticmethod
    def get_mixed_output(key, img_db_txn, txt_db, tokenizer=None):
        txt_data = load_single_txt(txt_db[key])
        # import ipdb; ipdb.set_trace()
        
        original_region_key = txt_data['img_fname'] + f'${txt_data["mix_index"]}'
        original_region = VizOutput.get_region_output(original_region_key, img_db_txn)
        
        mix_region = VizOutput(txt_data["mix_img_flk_id"])
        mix_region.draw_bounding_box(txt_data['mix_bb'])
        # original_region = VizOutput(fname2imgid(txt_data['img_fname'])+'.jpg')
        # original_region.draw_bounding_box(txt_data['mix_bb'])
        
        # mix_region_key = imgid2fname(txt_data["mix_img_flk_id"].replace('.jpg',''))
        # mix_region_key = mix_region_key + f'${txt_data["mix_index"]}'
        # mix_region = VizOutput.get_region_output(mix_region_key, img_db_txn)
        
        if tokenizer is not None:
            sentence = [tokenizer.convert_ids_to_tokens(txt_data['input_ids'])]
        else:
            sentence = [str(txt_data['input_ids'])]
        
        output_img = get_concat_h(original_region.img, mix_region.img, sentence)
        return output_img

def draw_bounding_box(img_name, img_bb, outline=(255, 0, 0, 255)):
    source_img = Image.open(origin_img_dir + img_name).convert("RGB")
    width, height = source_img.size

    draw = ImageDraw.Draw(source_img, 'RGBA')
    p1 = (width*img_bb[0], height*img_bb[1])
    p2 = (width*img_bb[2], height*img_bb[3])
    draw.rectangle((p1, p2), outline=outline, width=2)
    return source_img

def draw_region(key, img_db_txn):
    img_name, img_key, bb_idx = split_region_key(key)
    bb = load_single_img(img_db_txn, img_key)['norm_bb'][bb_idx]
    img = draw_bounding_box(img_name, bb)
    return img

def draw_example(img_db_txn, txt_db, key, out_dir='./sample', tag=None):
    txt_data = load_single_txt(txt_db[key])
    if 'mix_region' in txt_data:
        sub_region_key = txt_data['mix_region']


if __name__ == '__main__':
    name2nbb, img_db_txn = load_img_db('/data/share/UNITER/ve/img_db/flickr30k')
    txt_db = load_txt_db('/data/share/UNITER/ve/da/pos/seed2/GloVe/txt_db/ve_train.db')
    
    from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    tokenizer = None
    # import ipdb; ipdb.set_trace()
    # key = b'0_1000092795.jpg#1r1c'
    key = b'84108_2377496811.jpg#1r1e'
    img = VizOutput.get_mixed_output(key, img_db_txn, txt_db, tokenizer=tokenizer)
    img.save(f'{key}.jpeg', 'JPEG')

    # real_txt_db = TxtLmdb('/data/share/UNITER/ve/txt_db/ve_train.db')
    # fake_txt_db = TxtLmdb('/data/share/UNITER/ve/da/seed2/txt_db/ve_train.db')
    
    
    # real_db = real_txt_db.load_dict()
    # fake_db = fake_txt_db.load_dict()
    
    # real_db_keys = list(real_db.keys())
    # fake_db_keys = list(fake_db.keys())
    
    # fake2real = {}
    # for k in fake_db_keys:
    #     fake2real[k] = k.split(b'_')[1]
    
    # random.seed(42)
    # random.shuffle(fake_db_keys)
    # fake_db_keys_train, fake_db_keys_dev = fake_db_keys[:-1500], fake_db_keys[-1500:]
    # real_db_keys_train = [fake2real[k] for k in fake_db_keys_train]
    # real_db_keys_dev = [fake2real[k] for k in fake_db_keys_dev]
    
    # base = '/data/share/UNITER/ve/da/real_fake'
    # real_infos = real_txt_db.infos
    # fake_infos = fake_txt_db.infos
    
    # save_args = [
    #     # (fake_db_keys_train, fake_db, {'labels': [0], 'scores': [1.0]}, 'train/fake', fake_infos),
    #     # (fake_db_keys_dev, fake_db, {'labels': [0], 'scores': [1.0]}, 'dev/fake', fake_infos),
    #     (real_db_keys_train, real_db, {'labels': [1], 'scores': [1.0]}, 'train/real', real_infos),
    #     (real_db_keys_dev, real_db, {'labels': [1], 'scores': [1.0]}, 'dev/real', real_infos),
    # ]
    
    # # import ipdb; ipdb.set_trace()
    
    # for args in save_args:
    #     db = {}
    #     meta = args[4]['meta']
    #     id2len = args[4]['id2len']
    #     txt2img = args[4]['txt2img']
    #     img2txts = args[4]['img2txts']
        
    #     infos_out = {
    #         'meta': meta,
    #         'id2len': {},
    #         'txt2img': {},
    #         'img2txts': img2txts
    #     }
        
    #     for k in args[0]:
    #         v = load_single_txt(args[1][k])
    #         db[k] = {
    #             'input_ids': v['input_ids'],
    #             'img_fname': v['img_fname'],
    #             'target': args[2]
    #         }
    #         for kk in v.keys():
    #             if 'mix' in kk:
    #                 db[k][kk] = v[kk]
    #         infos_out['id2len'][k.decode('utf-8')] = id2len[k.decode('utf-8')]
    #         infos_out['txt2img'][k.decode('utf-8')] = txt2img[k.decode('utf-8')]
            
    #     # import ipdb; ipdb.set_trace()
        
    #     TxtLmdb.save_db(os.path.join(base, args[3]), db, infos_out)
        
        
    
