#!/bin/bash

model=transformer
train_file=dataset/kor_al.train
pool_file=dataset/kor_pool.train

for suffix in al; do
    for lang in kor; do

#        2594 28399 15102 506 27827
        # Loop to run the training 5 times
        for i in {1..25}; do
            python active-learning/difficulty_evaluator.py "${lang}_al.train" "${lang}.gold"

            for seed in 2594 28399 15102 506 27827; do

                bash active-learning/train.sh $lang $model $suffix $seed
                bash active-learning/ensemble_eval.sh $lang $model $suffix $seed smart

                # Rename .tsv files without a number in front to ...i.tsv
                for tsv_file in checkpoints/sig22/transformer/*[^0-9].tsv; do
                    if [ -f "$tsv_file" ]; then
                        base_name=$(basename "$tsv_file" .tsv)
                        mv "$tsv_file" "checkpoints/sig22/transformer/${base_name}_${i}_${seed}.tsv"
                    fi
                done

                bash active-learning/al_ensemble_eval.sh $lang $model $suffix $seed smart ensemble

                for tsv_file in checkpoints/sig22/transformer/*[^0-9].tsv; do
                    if [ -f "$tsv_file" ]; then
                        base_name=$(basename "$tsv_file" .tsv)
                        mv "$tsv_file" "checkpoints/sig22/transformer/${base_name}_pool_${i}_${seed}.tsv"
                    fi
                done

                # Delete model checkpoints and decode
                rm -f checkpoints/sig22/transformer/*epoch_[0-9]*
            done
            python active-learning/ensemble_predict.py "$i"
            python active-learning/ensemble_resample.py "$i" "$train_file" "$pool_file"
        done
    done
done
