import random
import os
import sys
sys.path.append(os.path.join(os.path.abspath(__file__+'/../..'))) # UNITER/

from data_aug.utils import (
    TxtLmdb,
    load_single_txt
)

def main():
    real_txt_db = TxtLmdb('/data/share/UNITER/ve/txt_db/ve_train.db')
    # fake_txt_db = TxtLmdb('/data/share/UNITER/ve/da/seed2/txt_db/ve_train.db')
    fake_txt_db = TxtLmdb('/data/share/UNITER/ve/da/simsub-seed42/txt_db/ve_train.db')
    base = '/data/share/UNITER/ve/da/real_fake/simsub'
    
    real_db = real_txt_db.load_dict()
    fake_db = fake_txt_db.load_dict()
    
    real_db_keys = list(real_db.keys())
    fake_db_keys = list(fake_db.keys())
    
    fake2real = {}
    for k in fake_db_keys:
        fake2real[k] = k.split(b'_')[1]
    
    random.seed(42)
    random.shuffle(fake_db_keys)
    fake_db_keys_train, fake_db_keys_dev = fake_db_keys[:-1500], fake_db_keys[-1500:]
    real_db_keys_train = [fake2real[k] for k in fake_db_keys_train]
    real_db_keys_dev = [fake2real[k] for k in fake_db_keys_dev]
    
    real_infos = real_txt_db.infos
    fake_infos = fake_txt_db.infos
    
    save_args = [
        # (fake_db_keys_train, fake_db, {'labels': [0], 'scores': [1.0]}, 'train/fake', fake_infos),
        # (fake_db_keys_dev, fake_db, {'labels': [0], 'scores': [1.0]}, 'dev/fake', fake_infos),
        (real_db_keys_train, real_db, {'labels': [1], 'scores': [1.0]}, 'train/real', real_infos),
        (real_db_keys_dev, real_db, {'labels': [1], 'scores': [1.0]}, 'dev/real', real_infos),
    ]
    
    for args in save_args:
        db = {}
        meta = args[4]['meta']
        id2len = args[4]['id2len']
        txt2img = args[4]['txt2img']
        img2txts = args[4]['img2txts']
        
        infos_out = {
            'meta': meta,
            'id2len': {},
            'txt2img': {},
            'img2txts': img2txts
        }
        
        for k in args[0]:
            v = load_single_txt(args[1][k])
            db[k] = {
                'input_ids': v['input_ids'],
                'img_fname': v['img_fname'],
                'target': args[2]
            }
            for kk in v.keys():
                if 'mix' in kk:
                    db[k][kk] = v[kk]
            infos_out['id2len'][k.decode('utf-8')] = id2len[k.decode('utf-8')]
            infos_out['txt2img'][k.decode('utf-8')] = txt2img[k.decode('utf-8')]
            
        # import ipdb; ipdb.set_trace()
        
        TxtLmdb.save_db(os.path.join(base, args[3]), db, infos_out)

if __name__ == '__main__':
    main()