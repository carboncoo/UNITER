import json
from os.path import abspath, dirname
from transformers import AutoTokenizer

path = '/data/share/UNITER/vqa/txt_db/vqa_train.db'

ans2label = json.load(open(f'{dirname(abspath(__file__))}'
                               f'/utils/ans2label.json'))

# tokenize
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# ans2tokid = {ans:tokenizer.encode(ans) for ans in ans2label.keys()}

# with open(f'{dirname(abspath(__file__))}/utils/ans2tokid2.json', 'w') as fout:
#   json.dump(ans2tokid, fout)

ans2tokid = json.load(open(f'{dirname(abspath(__file__))}'
                               f'/utils/ans2tokid2.json'))

# import ipdb; ipdb.set_trace()
# for ans, tok in ans2tokid.items():
#   dec = tokenizer.decode(tok[1:-1])
#   if ans != dec:
#     print(ans)
#     print(dec)

# for key1, key2 in zip(ans2label.keys(), ans2tokid.keys()):
#   if key1 != key2:
#     print(key1, key2)

# cnt = 0
# for tokid in ans2tokid.values():
#   if tokid == 100:
#     cnt += 1
# print(cnt)