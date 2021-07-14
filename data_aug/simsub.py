import os
import random
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from utils import (
    load_txt_db,
    load_img_db,
    load_single_img,
    load_single_region,
    load_single_txt
)

def create_class_region_mapping(img_db_txn, keys, n_classes=1601):
    region2class = {}
    class2region = [[] for i in range(1601)]
    for key in tqdm(keys):
        img = load_single_img(img_db_txn, key)
        soft_labels = img['soft_labels']
        max_values = soft_labels.max(axis=-1)
        max_idxs = soft_labels.argmax(axis=-1)
        region2class[key] = []
        for i, idx, value in zip(range(max_idxs.shape[0]), max_idxs, max_values):
            region2class[key].append((idx, value))
            if value < 0.5:
                continue
            new_key = f'{key}${i}'
            class2region[idx].append(new_key)
    
    with open('region2class.mp', 'wb') as f1, open('class2region.mp', 'wb') as f2:
        msgpack.dump(region2class, f1)
        msgpack.dump(class2region, f2)
    return region2class, class2region

origin_img_dir = '/data/share/UNITER/origin_imgs/flickr30k/flickr30k-images/'

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

def split_region_key(key):
    img_name = key.split('_')[-1].split('.')[0].lstrip('0') + '.jpg'
    img_key = key.split('$')[0]
    bb_idx = int(key.split('$')[-1])
    return img_name, img_key, bb_idx

def substitute_txt(original_txt):
    return original_txt
    
def substitute(img_db_txn, txt_db, class2region, out_dir, seed=42):
    random.seed(seed)
    txt_env_out = lmdb.open(out_dir, map_size=int(1e11))
    txt_txn_out = txt_env_out.begin(write=True)
    sample_cnt = 0
    for k, data in tqdm(txt_db.items()):
        txt_data = load_single_txt(data)
        img_fname = txt_data['img_fname']
        img_data = load_single_img(img_db_txn, img_fname)
        region_soft_labels = img_data['soft_labels']
        max_idx = np.unravel_index(region_soft_labels.argmax(), region_soft_labels.shape) # [row, col]
        candidates = class2region[max_idx[1]]
        sub_region_key = random.sample(candidates, 1)[0]
        sub_region_feat = load_single_region(img_db_txn, sub_region_key)
        
        txt_data['mix_index'] = max_idx[0]
        txt_data['mix_feature'] = sub_region_feat
        txt_data['mix_region'] = sub_region_key
        
        mix_txt_key = str(sample_cnt) + '_' + k.decode('utf-8')
        txt_txn_out.put(mix_txt_key.encode('utf-8'), compress(msgpack.dumps(txt_data, use_bin_type=True)))
        
        if sample_cnt % 1000 == 0:
            txt_txn_out.commit()
            txt_txn_out = txt_env_out.begin(write=True)
        
        sample_cnt += 1
    
    txt_txn_out.commit()
    txt_env_out.close()

def sample(img_db_txn, mixed_txt_db, sample_n=10, out_dir='./sample'):
    keys = mixed_txt_db.keys()
    sampled_keys = random.sample(keys, sample_n)
    os.makedirs(out_dir, exist_ok=True)
        
    for key in sampled_keys:
        txt_data = load_single_txt(mixed_txt_db[key])
        sentence = ' '.join(txt_data['toked_hypothesis'])
        
        original_region_key = txt_data['img_fname'] + f"${txt_data['mix_index']}"
        sub_region_key = txt_data['mix_region']
        original_region = draw_region(original_region_key, img_db_txn)
        sub_region = draw_region(sub_region_key, img_db_txn)
        
        output_img = get_concat_h(original_region, sub_region, [sentence])
        output_img.save(os.path.join(out_dir, f'{key}.jpeg'), 'JPEG')

if __name__ == '__main__':
    # name2nbb, img_db_txn = load_img_db('/data/share/UNITER/ve/img_db/flickr30k')
    # txt_db = load_txt_db('/data/share/UNITER/ve/txt_db/ve_train.db')
    
    # CREATE
    # region2class, class2region = create_class_region_mapping(img_db_txn, list(name2nbb.keys())) # it takes about 35 seconds
    
    # LOAD
    # with open('region2class.mp', 'rb') as f1, open('class2region.mp', 'rb') as f2:
    #     region2class = msgpack.load(f1)
    #     class2region = msgpack.load(f2)

    # SUBSTITUTE
    # seed = 42
    # out_dir = f'/data/share/UNITER/ve/da/simsub-seed{seed}-1/txt_db/ve_train.db'
    # substitute(img_db_txn, txt_db, class2region, out_dir, seed=seed)
    
    # SAMPLE
    name2nbb, img_db_txn = load_img_db('/data/share/UNITER/ve/img_db/flickr30k')
    seed = 42
    out_dir = f'/data/share/UNITER/ve/da/simsub-seed{seed}-1/txt_db/ve_train.db'
    txt_db = load_txt_db(out_dir)
    sample(img_db_txn, txt_db)
    # import ipdb; ipdb.set_trace()