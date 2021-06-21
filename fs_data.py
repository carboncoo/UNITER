import lmdb
import json
import random
import os
from os.path import exists


db_dir = "/data/share/UNITER/ve_fewshot/txt_db/ve_dev.db/seed_1"
output_db_dir = "/data/share/UNITER/ve_fewshot/txt_db/ve_train.db"
seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
num_of_sample = 32

def main():
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


if __name__ == "__main__":
	main()