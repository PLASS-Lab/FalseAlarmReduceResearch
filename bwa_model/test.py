import argparse
import os,sys,json 
import pandas as pd 
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score,f1_score,precision_score

from models.model_sentence import ModelVul
from datasets.dataset import DatasetTraining, DatasetTesting
from datasets.collated_functions import test_collated_fn, train_collated_fn



parser = argparse.ArgumentParser(description='Testing Vul-Bert')
parser.add_argument('--csv_path', type=str, default="", 
                    help='Path to file test_csv')
parser.add_argument('--num_classes', type=int, default=1, 
                    help='number of class [number of error types]')
parser.add_argument('--model_name_or_path', type=str, default="somemodel_checkpoints",
                    help='name of model')          
parser.add_argument('--ckpt', type=str, default="somemodel_checkpoints",
                    help='ckpt of model')        
parser.add_argument('--save_predict', type=str, default="./ok.csv",
                    help='save results to ')
parser.add_argument('--batch_size', type=int, default=8,
                    help='number of batch_size training') 
args = parser.parse_args()
device = "cpu"
if torch.cuda.is_available():
    device="cuda"

model = ModelVul(args.model_name_or_path, args.num_classes)
model=model.to(device)
model.load_state_dict(torch.load(args.ckpt, map_location=device))
optimizer  =torch.optim.Adam(model.parameters(), lr=1e-5)

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

val_pandas = pd.read_csv(args.csv_path)
val_dataset = DatasetTesting(val_pandas, tokenizer)
val_loader = torch.utils.data.DataLoader(val_dataset, 
                                            batch_size=args.batch_size, 
                                            collate_fn=test_collated_fn, 
                                            shuffle=False, num_workers=2)

print("-"*50)
print(" "*20,"Testing Model",sep="" )
print("-"*50)
reduce_value=[]
model.eval()
for data in tqdm(val_loader):
    keys = data.keys()
    for k in keys:
        if "tensor_" in k:
            data[k] = data[k].to(device)
    with torch.no_grad():
        out=model(data)
    out=out.softmax(-1).cpu().numpy()
    for i in range(out.shape[0]):
        prob_lines=[100,] * len(data['source_code_lines_raw'][i])
        class_lines=[0,] * len(data['source_code_lines_raw'][i])
        for index,value in enumerate(data['mapping_line_to_line_in_source_raw'][i]):
        # print(value)
            prob_lines[value] = int(100*out[i,index].max())
            class_lines[value] = int(out[i,index].argmax(-1))
        reduce_value.append({
            "prob_lines":prob_lines,
            "class_lines":class_lines,
            "source_code_lines_raw":data['source_code_lines_raw'][i]
        })
    # break
import json
print(reduce_value[0])
with open(args.save_predict,"w+") as f:
    json.dump(reduce_value,f)
# pd.DataFrame(reduce_value).to_csv(args.save_predict, index=False)