#!/bin/bash

# In this example, we show how to train DCPCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name_or_path bert-base-chinese \
    --train_file datasets/train.txt \
    --output_dir result/CHEF_train.ckpt \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 3e-2 \
    --max_seq_length 256 \
    --evaluation_strategy steps \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --pre_seq_len 16 \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
#    --do_eval \
#    --metric_for_best_model stsb_spearman \
#    --load_best_model_at_end \

