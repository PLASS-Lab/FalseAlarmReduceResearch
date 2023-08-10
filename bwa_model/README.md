#  BERT 모델을 이용한 라인별로 보안약점을 분석 모델 (BWA): BERT 모델을 이용한 라인별 보안약점 분석 모델(BWA)은 입력한 C/C++ 소스코드를 토큰화하고 임베딩한 후에 보안약점 패턴을 학습한 후에 분석하게 된다.

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

    

