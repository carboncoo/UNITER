# TXT_DB=/data/share/UNITER/ve/txt_db/ve_test.db # 0.9530325156695156
# TXT_DB=/data/share/UNITER/ve/da/seed2/txt_db/ve_train.db # 0.036324868250344165 (529527)
# TXT_DB=/data/share/UNITER/ve/da/threshold/0.900000/seed2/GloVe/txt_db/ve_train.db # 0.30923251686776004 (307154)
# TXT_DB=/data/share/UNITER/ve/da/threshold/0.85/seed2/GloVe/txt_db/ve_train.db # 0.30615983698726806 (368358)
# TXT_DB=/data/share/UNITER/ve/da/threshold/0.80/seed2/GloVe/300k/txt_db/ve_train.db # 0.305400171044 (334434)
# TXT_DB=/data/share/UNITER/ve/da/threshold/0.80/seed2/GloVe/500k/txt_db/ve_train.db # 0.30426455522888063 (500000)
# TXT_DB=/data/share/UNITER/ve/da/pos/seed2/GloVe/txt_db/ve_train.db # 0.31241854731405827 (359185)
# TXT_DB=/data/share/UNITER/ve/da/seed3/txt_db/ve_train.db # 0.056164860989146914 (529527)
TXT_DB=/data/share/UNITER/ve/da/simsub-seed42/txt_db/ve_train.db
IMG_DB=/data/share/UNITER/ve/img_db/flickr30k 

OUTPUT_DIR=/data/private/cc/experiment/MMP/UNITER/results/exp_results/real-fake-classify-99434aa392d9480484d8ae5934e353d7

horovodrun -np 1 python scorer.py \
--txt_db $TXT_DB \
--img_db $IMG_DB \
--output_dir $OUTPUT_DIR \
--checkpoint 4000 \
--batch_size 16384 \
--pin_mem \
--fp16