#!/bin/bash

drops=('0.1' '0.06' '0.07' '0.08' '0.09' '0.05' '0.11' '0.12' '0.13')
atdrops=('0.05' '0.06' '0.07' '0.08' '0.09' '0.1' '0.11' '0.12' '0.13')
for dr in ${drops[@]}
do
  for atdr in ${atdrops[@]}
  do
    python main.py --seed 5 --lr 2e-5 --weight_decay 0.0004 --dropout $dr --num_iters 12 --max_len 85 \
    --attention_probs_dropout_prob $atdr
  done
done