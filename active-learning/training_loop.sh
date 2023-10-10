#!/bin/bash

model=transformer

for suffix in al; do
    for lang in vep; do
        # Backup the initial training and pool data
        cp "dataset/${lang}_al.train" "dataset/${lang}_al.train_backup"
        cp "dataset/${lang}_pool.train" "dataset/${lang}_pool.train_backup"

#        2594 28399 15102 506 27827
        # Loop to run the training 5 times
        for seed in 2594 28399 15102 506 27827; do
            # Restore the density data from the backup at the beginning of each seed iteration
            cp "active-learning/${lang}_density_backup.tsv" "active-learning/${lang}_density.tsv"

            # Restore the training and pool data to their initial states
            cp "dataset/${lang}_al.train_backup" "dataset/${lang}_al.train"
            cp "dataset/${lang}_pool.train_backup" "dataset/${lang}_pool.train"

            for i in {1..25}; do
                python active-learning/difficulty_evaluator.py "${lang}_al.train" "${lang}.gold"

                bash active-learning/train.sh $lang $model $suffix $seed
                bash active-learning/eval.sh $lang $model $suffix $seed smart

                # Rename .tsv files without a number in front to ...i.tsv
                for tsv_file in checkpoints/sig22/transformer/*[^0-9].tsv; do
                    if [ -f "$tsv_file" ]; then
                        base_name=$(basename "$tsv_file" .tsv)
                        mv "$tsv_file" "checkpoints/sig22/transformer/${base_name}_${i}.tsv"
                    fi
                done

                bash active-learning/al_eval.sh $lang $model $suffix $seed smart information_density

                # Delete model checkpoints and decode
                rm -f checkpoints/sig22/transformer/*epoch_[0-9]*
            done
            # Add the seed as prefix to all the files that don't start with a number
            for file in checkpoints/sig22/transformer/*; do
                filename=$(basename "$file")
                if [[ ! "$filename" =~ ^[0-9] ]]; then
                    new_name="checkpoints/sig22/transformer/${seed}_${filename}"
                    mv "$file" "$new_name"
                fi
            done
        done

        # Optionally, remove the backup files after all seeds are processed
        rm "dataset/${lang}_al.train_backup"
        rm "dataset/${lang}_pool.train_backup"
    done
done
