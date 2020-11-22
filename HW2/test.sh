dp=('0.03' '0.05' '0.08' '0.1' '0.12' '0.16' '0.15')
for w in ${dp[@]}
do
  python main.py --seed 8 --lr 2e-5 --weight_decay 0.0004 --dropout $w --num_iters 15 --max_len 85
done