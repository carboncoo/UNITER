import os
import json
import shutil
import sys
sys.path.append(os.path.join(os.path.abspath(__file__+'/../..'))) # UNITER/

from data_aug.utils import (
    load_img_db,
    load_txt_db,
    VizOutput
)

all_dbs = [
    ('vanilla', '/data/share/UNITER/ve/da/seed2/txt_db/ve_train.db'),
    ('th-0.9', '/data/share/UNITER/ve/da/threshold/0.900000/seed2/GloVe/txt_db/ve_train.db'),
    ('th-0.85', '/data/share/UNITER/ve/da/threshold/0.85/seed2/GloVe/txt_db/ve_train.db'),
    ('th-0.8-300k', '/data/share/UNITER/ve/da/threshold/0.80/seed2/GloVe/300k/txt_db/ve_train.db'),
    ('th-0.8-500k', '/data/share/UNITER/ve/da/threshold/0.80/seed2/GloVe/500k/txt_db/ve_train.db'),
    ('pos', '/data/share/UNITER/ve/da/pos/seed2/GloVe/txt_db/ve_train.db')
]

def make_new_dirs(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)

def main():
    output_dir = '/data/share/UNITER/ve/da/case_study'
    name2nbb, img_db_txn = load_img_db('/data/share/UNITER/ve/img_db/flickr30k')
    top_k = 10
    
    for tag, db in all_dbs:
        print(tag, db)
        txt_db = load_txt_db(db)
        example_scores = json.load(open(os.path.join(db, 'results_test/results_4000_all.json')))
        example_scores = sorted(example_scores, key=lambda x: float(x["answer"]), reverse=True)

        cur_dir = os.path.join(output_dir, tag)
        make_new_dirs(cur_dir)
        
        with open(os.path.join(cur_dir, 'scores'), 'w') as fout:
            for example in example_scores:
                score = example['answer']
                idx = example['question_id']
                fout.write(f'{idx}\t{score}\n')
        
        case_scores = example_scores[:top_k]
        cur_best_dir = os.path.join(cur_dir, 'best')
        make_new_dirs(cur_best_dir)
        for case in case_scores:
            score = case['answer']
            idx = case['question_id']
            case_img = VizOutput.get_mixed_output(idx.encode('utf-8'), img_db_txn, txt_db, tokenizer=None)
            case_img.save(os.path.join(cur_best_dir, idx+'.jpeg'), 'JPEG')
        
        case_scores = example_scores[-top_k:]
        cur_worst_dir = os.path.join(cur_dir, 'worst')
        make_new_dirs(cur_worst_dir)
        for case in case_scores:
            score = case['answer']
            idx = case['question_id']
            case_img = VizOutput.get_mixed_output(idx.encode('utf-8'), img_db_txn, txt_db, tokenizer=None)
            case_img.save(os.path.join(cur_worst_dir, idx+'.jpeg'), 'JPEG')

if __name__ == '__main__':
    main()