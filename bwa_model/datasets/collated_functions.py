import torch 
import numpy as np 


def train_collated_fn(all_sample):
    raw_reduce={}
    for k in all_sample[0]:
        raw_reduce[k] = [all_sample[0][k]]
    for item in all_sample[1:]:
        for k in item:
          raw_reduce[k].append(item[k])
    max_len_sentence = max([len(i) for i in raw_reduce['source_code_line_tokenizers']])
    max_length_input_ids = max([len(i) for i in raw_reduce['input_ids']]) 
    max_length_input_ids=min(max_length_input_ids, 500)
    max_trix_sent = np.zeros((len(all_sample),max_length_input_ids,max_len_sentence)) 
    label = np.zeros((len(all_sample), max_len_sentence))
    input_batch_ids=[]
    for sample_index in range(len(all_sample)):
        input_ids = all_sample[sample_index]['input_ids']
        if len(input_ids) > max_length_input_ids:
            input_ids=input_ids[:max_length_input_ids]
        for i in range(len(input_ids)):
            line_index = all_sample[sample_index]['mapping_token_to_line_source_tokenizer'][i]
            max_trix_sent[sample_index,i,line_index] = 1.
        


        if len(input_ids) < max_length_input_ids:
            input_ids.extend([1,] * (max_length_input_ids - len(input_ids))) 
        label_sentent=all_sample[sample_index]['label_sentence_level']

        for k in range(len(label_sentent)):
            if label_sentent[k] != 0:
                label[sample_index, k] = label_sentent[k]

    
        input_batch_ids.append(input_ids)

    input_batch_ids=torch.tensor(input_batch_ids).long()
    label=torch.from_numpy(label).long()
    max_trix_sent=torch.from_numpy(max_trix_sent).float()

    raw_reduce['tensor_matrix_sentent']=max_trix_sent
    raw_reduce['tensor_label']=label
    raw_reduce['tensor_input_batch_ids'] = input_batch_ids
    return raw_reduce 


def test_collated_fn(all_sample):
    raw_reduce={}
    for k in all_sample[0]:
        raw_reduce[k] = [all_sample[0][k]]
    for item in all_sample[1:]:
        for k in item:
            raw_reduce[k].append(item[k])
    max_len_sentence = max([len(i) for i in raw_reduce['source_code_line_tokenizers']])
    max_length_input_ids = max([len(i) for i in raw_reduce['input_ids']]) 
    max_length_input_ids=min(max_length_input_ids, 500)
    max_trix_sent = np.zeros((len(all_sample),max_length_input_ids,max_len_sentence)) 
    input_batch_ids=[]
    for sample_index in range(len(all_sample)):
        input_ids = all_sample[sample_index]['input_ids']
        if len(input_ids) > max_length_input_ids:
            input_ids=input_ids[:max_length_input_ids]
        for i in range(len(input_ids)):
            line_index = all_sample[sample_index]['mapping_token_to_line_source_tokenizer'][i]
            max_trix_sent[sample_index,i,line_index] = 1.
        


        if len(input_ids) < max_length_input_ids:
            input_ids.extend([1,] * (max_length_input_ids - len(input_ids))) 
        input_batch_ids.append(input_ids)

    input_batch_ids=torch.tensor(input_batch_ids).long()
    max_trix_sent=torch.from_numpy(max_trix_sent).float()

    raw_reduce['tensor_matrix_sentent']=max_trix_sent
    raw_reduce['tensor_input_batch_ids'] = input_batch_ids
    return raw_reduce 