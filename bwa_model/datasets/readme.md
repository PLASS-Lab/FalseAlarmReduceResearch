
# Training

python3 train.py --train_csv_path=example_datasets/old/train.csv \
                  --val_csv_path=example_datasets/old/val.csv \
                  --num_classes=120 \
                  --do_train \
                  --model_name_or_path="neulab/codebert-cpp" \
                  --epoch=10 \
                  --batch_size=16


# Testing

python3 test.py --csv_path=example_datasets/test.csv\
                --num_classes=120\
                --model_name_or_path='neulab/codebert-cpp'\
                --save_predict="./pred.json"\
                --ckpt=model.ckpt\
                --batch_size=64

Output testing la 1 file json: [
    {
        "prob_lines":prob_lines,
        "class_lines":class_lines,
        "source_code_lines_raw":source_code_lines_raw
    }
]
# Pharse 2

Ghep output pharse 1 + cac tools:

1. Chuan bi file csv co dang: [score_deep, score_tool1, score_tool2,...]
    

