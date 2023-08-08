
# Training

python3 train.py --train_csv_path=dataset/train.csv \
                  --val_csv_path=dataset/val.csv \
                  --num_classes=120 \
                  --do_train \
                  --model_name_or_path="neulab/codebert-cpp" \
                  --epoch=10 \
                  --batch_size=16


# Testing

python3 test.py --csv_path=dataset/test.csv\
                --num_classes=120\
                --model_name_or_path='neulab/codebert-cpp'\
                --save_predict="./pred.json"\
                --ckpt=model.ckpt\
                --batch_size=64

Output testing: file json: [
    {
        "prob_lines":prob_lines,
        "class_lines":class_lines,
        "source_code_lines_raw":source_code_lines_raw
    }
]

    

