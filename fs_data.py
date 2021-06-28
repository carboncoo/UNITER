# encoding=utf-8
import lmdb
import json
import pickle
import random
import os
from os.path import exists, abspath, dirname
from data import (TxtTokLmdb, ImageLmdbGroup, ConcatDatasetWithLens,
                  VqaDataset, VqaEvalDataset,
                  vqa_collate, vqa_eval_collate)

from lz4.frame import compress, decompress
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
num_of_sample = 32
db_dir = "/data/share/UNITER/vqa/txt_db/vqa_train.db"
output_db_dir = "/data/share/UNITER/vqa_fewshot/txt_db/vqa_train.db"

def ve():
    # import ipdb; ipdb.set_trace()
    
    # read from db
    env_in = lmdb.open(db_dir)
    txn_in = env_in.begin()
    db = {}
    for key, value in txn_in.cursor():
        db[key] = value
    print('db length:', len(db))
    env_in.close()

    id2len = json.load(open(db_dir + '/id2len.json'))
    img2txts = json.load(open(db_dir + '/img2txts.json'))
    txt2img = json.load(open(db_dir + '/txt2img.json'))
    print(len(id2len))
    print(len(img2txts))
    print(len(txt2img))

    # print(seeds)
    # for s in seeds:
    #     print("sampling by seed %d"%(s))
    #     random.seed(s)
    #     # random sample
    #     out_img2txts = dict(random.sample(img2txts.items(), num_of_sample))
    #     txts = [txt for sublist in out_img2txts.values() for txt in sublist]
    #     out_id2len = {k: id2len[k] for k in txts}
    #     out_txt2img = {k: txt2img[k] for k in txts}
    #     out_db = {k.encode('utf-8'): db[k.encode('utf-8')] for k in txts}
    #     # print(out_id2len)
    #     # print(out_txt2img)
    #     # print(out_db)
        
    #     # write to db
    #     output_dir = output_db_dir + '/seed_%d'%(s)
    #     if not exists(output_dir):
    #         os.makedirs(output_dir)
    #     else:
    #         raise ValueError('Found existing DB. Please explicitly remove '
    #                         'for re-processing')

    #     env_out = lmdb.open(output_dir)
    #     txn_out = env_out.begin(write=True)
    #     for k, v in out_db.items():
    #         txn_out.put(k, v)
    #     txn_out.commit()
    #     env_out.close()

    #     json.dump(out_id2len, open(output_dir + '/id2len.json', 'w'))
    #     json.dump(out_img2txts, open(output_dir + '/img2txts.json', 'w'))
    #     json.dump(out_txt2img, open(output_dir + '/txt2img.json', 'w'))

def vqa():
    # import ipdb; ipdb.set_trace()
    
    # read from db
    env_in = lmdb.open(db_dir)
    txn_in = env_in.begin()
    db = {}
    for key, value in txn_in.cursor():
        db[key] = value
    print('db length:', len(db))
    env_in.close()

    meta = json.load(open(db_dir + '/meta.json'))
    id2len = json.load(open(db_dir + '/id2len.json'))
    img2txts = json.load(open(db_dir + '/img2txts.json'))
    txt2img = json.load(open(db_dir + '/txt2img.json'))
    ans2label = json.load(open(f'{dirname(abspath(__file__))}'
                               f'/utils/ans2label.json'))
    # ans2label = pickle.load(open(db_dir + '/ans2label.pkl'))
    print(len(id2len))
    print(len(img2txts))
    print(len(txt2img))
    print(len(ans2label))

    label2ans = {v:k for k,v in ans2label.items()}
    answer_cnt = {ans: 0 for ans in ans2label.keys()}

    # for v in db.values():
    #     item = msgpack.loads(decompress(v), raw=False)
    #     answers = item['answers']
    #     target = item['target']
    #     # print(target)
    #     for label in target['labels']:
    #         answer_cnt[label2ans[label]] += 1
            
    # answer_cnt = json.load(open(f'{dirname(abspath(__file__))}'
    #                            f'/utils/ans_cnt.json'))
    # print(len(answer_cnt))
    # json.dump(answer_cnt, open(f'{dirname(abspath(__file__))}'
    #                            f'/utils/ans_cnt.json', 'w'))
    
    # print(seeds)
    for s in seeds:
        print("sampling by seed %d"%(s))
        random.seed(s)
        # random sample
        out_img2txts = dict(random.sample(img2txts.items(), num_of_sample))
        txts = [txt for sublist in out_img2txts.values() for txt in sublist]
        out_id2len = {k: id2len[k] for k in txts}
        out_txt2img = {k: txt2img[k] for k in txts}
        out_db = {k.encode('utf-8'): db[k.encode('utf-8')] for k in txts}
        # print(out_id2len)
        # print(out_txt2img)
        # print(out_db)

        # statistic
        answer_cnt = {ans: 0 for ans in ans2label.keys()}

        for v in out_db.values():
            item = msgpack.loads(decompress(v), raw=False)
            answers = item['answers']
            target = item['target']
            # print(target)
            for label in target['labels']:
                answer_cnt[label2ans[label]] += 1
        output_dir = output_db_dir + '/seed_%d'%(s)
        # json.dump(answer_cnt, open(output_dir + '/ans_cnt.json', 'w'))
        num_of_label = {}
        for cnt in answer_cnt.values():
            if cnt in num_of_label:
                num_of_label[cnt] += 1
            else:
                num_of_label[cnt] = 1
        print(num_of_label)
        # write to db
        # output_dir = output_db_dir + '/seed_%d'%(s)
        # if not exists(output_dir):
        #     os.makedirs(output_dir)
        # else:
        #     raise ValueError('Found existing DB. Please explicitly remove '
        #                     'for re-processing')

        # env_out = lmdb.open(output_dir)
        # txn_out = env_out.begin(write=True)
        # for k, v in out_db.items():
        #     txn_out.put(k, v)
        # txn_out.commit()
        # env_out.close()

        # json.dump(out_id2len, open(output_dir + '/id2len.json', 'w'))
        # json.dump(out_img2txts, open(output_dir + '/img2txts.json', 'w'))
        # json.dump(out_txt2img, open(output_dir + '/txt2img.json', 'w'))
        # json.dump(meta, open(output_dir + '/meta.json', 'w'))

if __name__ == "__main__":
	# ve()
    vqa()