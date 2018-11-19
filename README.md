# machine-translation

 python train.py --data id2/ --source-lang jp --target-lang en --lr 0.01 --clip-norm 1 --max-tokens 10000 --save-dir checkpoints/lstm36
 
 
python generate.py  --checkpoint-path checkpoints/lstm36/checkpoint_last.pt --data idata --output ab  --beam-size 12 --unk-penalty 100 --normalize_scores True --max-len 25

cat output | perl  multi-bleu.perl -lc idata/test.en
