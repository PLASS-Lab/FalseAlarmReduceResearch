import argparse
import os,sys,json 
import pandas as pd 
from tqdm import tqdm
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score,f1_score,precision_score
from models.model_sentence import ModelVul
from models.focal_loss import FocalLoss
from datasets.dataset import DatasetTraining, DatasetTesting
from datasets.collated_functions import test_collated_fn, train_collated_fn

logger = logging.getLogger(__name__)

def test(args, model):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = args.device
    test_pandas = pd.read_csv(args.csv_path)
    test_dataset = DatasetTesting(test_pandas, tokenizer)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size=args.batch_size, 
                                                collate_fn=test_collated_fn, 
                                                shuffle=False, num_workers=2)
    logger.info("***** Running testing *****")
    logger.info("  Num examples = %d", len(test_loader))
    reduce_value=[]
    for data in tqdm(test_loader):
        keys = data.keys()
        for k in keys:
            if "tensor_" in k:
                data[k] = data[k].to(device)
        with torch.no_grad():
            out=model(data)
        out=out.softmax(-1).cpu().numpy()
        
        for i in range(out.shape[0]):
            prob_lines=[100000,] * len(data['source_code_lines_raw'][i])
            class_lines=[0,] * len(data['source_code_lines_raw'][i])
            for index,value in enumerate(data['mapping_line_to_line_in_source_raw'][i]):
                prob_lines[value] = int(100000*out[i,index].max())
                class_lines[value] = int(out[i,index].argmax(-1))
            reduce_value.append({
                "prob_lines":prob_lines,
                "class_lines":class_lines,
                "source_code_lines_raw":data['source_code_lines_raw'][i]
            })
    logger.info("  Num reduce value = %d", len(reduce_value))        
    logger.info(f"  Frist reduce_value: {reduce_value[0]}")  
    logger.info(f"  Save path: {args.save_predict}")        
    with open(args.save_predict,"w+") as f:
        json.dump(reduce_value,f)
    
def train(args, model):
    """ Train the model """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = args.device
    
    train_pandas = pd.read_csv(args.train_csv_path)
    val_pandas = pd.read_csv(args.val_csv_path)

    train_dataset = DatasetTraining(train_pandas, tokenizer)
    val_dataset = DatasetTraining(val_pandas, tokenizer)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=args.batch_size, 
                                                collate_fn=train_collated_fn,
                                                shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                                batch_size=args.batch_size, 
                                                collate_fn=train_collated_fn, 
                                                shuffle=True, num_workers=2)
    
    torch.save(model.state_dict(),"model.ckpt")
    
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(val_loader))
    logger.info("  Num Epochs = %d", 10)
    logger.info("  Total train batch size = = %d", 8)
    
    for epoch in range(args.epoch):
        print(f"Training epoch: {epoch}")
        logger.info(f" Training epoch: {epoch}")  
        pred=[]
        label=[]
        total_loss=0.
        for data in tqdm(train_loader, total=len(train_loader)):
            keys = data.keys()
            for k in keys:
                if "tensor_" in k:
                    data[k] = data[k].to(device)
            optimizer.zero_grad()
            out=model(data)
            loss=torch.nn.CrossEntropyLoss()(
                out.view(out.size(0)*out.size(1),-1),
                data['tensor_label'].view(-1,)
            )
            loss.backward()
            pred.append(out.detach().cpu().reshape(-1, out.size(-1)))
            label.append(data['tensor_label'].detach().cpu().reshape(-1,))
            total_loss=total_loss+loss.detach().cpu().item()
            optimizer.step()
        pred=torch.argmax(torch.cat(pred,0),-1).reshape(-1,).numpy()
        label=torch.cat(label,0).reshape(-1,).numpy()
        acc=accuracy_score(label,pred)
        f1=f1_score(label, pred,  average='macro')
        p=precision_score(label, pred,  average='macro')
        print(f"Train acc: {acc}\nTrain F1: {f1}\nTrain Pr: {p}\n")

        print(f"Validation epoch {epoch}")
        pred=[]
        label=[]
        total_loss=0.
        for data in tqdm(val_loader, total=len(val_loader)):
            keys = data.keys()
            for k in keys:
                if "tensor_" in k:
                    data[k] = data[k].to(device)

            with torch.no_grad():
                out=model(data)

            loss=torch.nn.CrossEntropyLoss()(
                out.view(out.size(0)*out.size(1),-1),
                data['tensor_label'].view(-1,)
            )
            pred.append(out.detach().cpu().reshape(-1, out.size(-1)))
            label.append(data['tensor_label'].detach().cpu().reshape(-1,))
            total_loss=total_loss+loss.detach().cpu().item()
            
        pred=torch.argmax(torch.cat(pred,0),-1).reshape(-1,).numpy()
        label=torch.cat(label,0).reshape(-1,).numpy()
        acc=accuracy_score(label,pred)
        f1=f1_score(label, pred,  average='macro')
        p=precision_score(label, pred,  average='macro')
        print(f"Val acc: {acc}\nVal F1: {f1}\nVal Pr: {p}\n")
        logger.info(f"  acc: {acc}")  
        logger.info(f"  f1: {f1}")
        logger.info(f"  precision score: {p}")    
    print("Saving model to model.ckpt")
    torch.save(model.state_dict(),"model.ckpt")
    
def main():
    parser = argparse.ArgumentParser(description='Training Vul-Bert')
    parser.add_argument("--do_train", action='store_true',
                      help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument('--train_csv_path', type=str, default="", 
                        help='Path to file train_csv')
    parser.add_argument('--val_csv_path', type=str, default="",
                        help='Path to file val_csv')
    parser.add_argument('--num_classes', type=int, default=1, 
                        help='number of class [number of error types]')
    parser.add_argument('--model_name_or_path', type=str, default="somemodel_checkpoints",
                        help='name of model or name of checkpoint to finetune')          
    parser.add_argument('--epoch', type=int, default=10,
                        help='number of epochs training')              
    parser.add_argument('--batch_size', type=int, default=8,
                        help='number of batch_size training')         
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S', filename='train.log', level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    model = ModelVul(args.model_name_or_path, args.num_classes)
    model=model.to(device)
    
    logger.info("Training/evaluation parameters %s", args)
    if args.do_train:
        train(args, model)
    if args.do_test:
        model_path = f'model.bin'
        model.load_state_dict(torch.load(model_path))
        test(model)
      
if __name__ == "__main__":
    main()