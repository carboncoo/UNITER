ROOT=/data/private/mxy
CODE=$ROOT/code/cc-UNITER
CKPT=$ROOT/exp/UNITER/pretrained
EXP=$ROOT/exp/UNITER/finetune
IMG_DB=/data/share/UNITER/img_db
TXT_DB=/data/share/UNITER/txt_db
export CUDA_VISIBLE_DEVICES=1

# train nlvr2
# CUDA_VISIBLE_DEVICES=5 python train_nlvr2.py --config config/train-softprompt-nlvr2-base-1gpu.json


# train vcr
VCR_EXP=$EXP/vcr/default
# nohup horovodrun -np 4 python train_vcr.py --config config/train-vcr-base-4gpu.json \
#     --output_dir $VCR_EXP > vcr.out 2>&1 &


# train ve
# VE_EXP=$EXP/ve_fewshot/seed_1
# horovodrun -np 2 python train_ve.py --config config/train-ve-fs-base-2gpu.json \
#     --output_dir $VE_EXP &

# few-shot finetune
# VE_EXP=$EXP/ve_fewshot/seed_5
# python train_ve.py --config config/train-ve-fs-base-1gpu.json \
#     --output_dir $VE_EXP

# few-shot prompt
VE_DATA_SEED=3
VE_EXP=$EXP/ve_fewshot_prompt/seed_${VE_DATA_SEED}
# TOKENIZERS_PARALLELISM=true python train_ve.py --config config/train-ve-fs-prompt-base-1gpu.json \
#     --output_dir $VE_EXP \
#     --train_txt_db /data/share/UNITER/ve_fewshot/txt_db/ve_train.db/seed_$DATA_SEED \
#     --val_txt_db /data/share/UNITER/ve_fewshot/txt_db/ve_dev.db/seed_$DATA_SEED



# zero-shot itr
NGPU=6
ZS_ITM_RESULT=$EXP/itr/zs_result
# horovodrun -np $NGPU python inf_itm.py \
#     --txt_db $TXT_DB/itm_flickr30k_test.db --img_db $IMG_DB/flickr30k \
#     --checkpoint $CKPT/uniter-base.pt --model_config $CODE/config/uniter-base.json \
#     --output_dir $ZS_ITM_RESULT --fp16 --pin_mem

# itr normal finetune
# nohup horovodrun -np 4 python train_itm.py --config config/train-itm-flickr-base-8gpu.json > itr_flickr_normal.out 2>&1 &


# train vqa
VQA_EXP=$EXP/vqa/finetune
# python train_vqa.py --config config/train-vqa-base-1gpu.json \
#     --output_dir $VQA_EXP

# prompt
VQA_EXP=$EXP/vqa/prompt
# TOKENIZERS_PARALLELISM=true python train_vqa.py --config config/train-vqa-prompt-base-1gpu.json \
#     --output_dir $VQA_EXP

# few-shot finetune
VQA_DATA_SEED=2
VQA_EXP=$EXP/vqa/fewshot/finetune/seed_${VQA_DATA_SEED}
python train_vqa.py --config config/train-vqa-fs-base-1gpu.json \
    --output_dir $VQA_EXP
    --train_txt_db [/data/share/UNITER/vqa_fewshot/txt_db/vqa_train.db/seed_${VQA_DATA_SEED}] \
    --val_txt_db /data/share/UNITER/vqa/txt_db/vqa_devval.db