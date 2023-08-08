import pandas as pd 
import os,sys,json,ast


def remove_comment(source_code_lines):
    """source_code_lines: List object
        - List[line_code]
    """
    type_line=[]
    content=[line for line in source_code_lines] # copy object source_code_lines
    for line in content: 
      if "/*" in line and "*/" in line:
        type_line.append(0)
        continue
      if line[:2] == '/*':
        type_line.append(1)
        continue
      if line[-2:]=="*/":
        type_line.append(2) 
        continue
      type_line.append(-1) 
    start_line=[]
    end_line=[]
    for index in range(len(content)):
      type_in_line=type_line[index]
      line=content[index]
      if type_in_line==-1:
        continue 
      if type_in_line==0:
        starts=[]
        ends = []
        # print()
        for k in range(len(line)-1):
          if line[k] == '/' and line[k+1] =='*':
            starts.append(k) 
          if line[k] == '*' and line[k+1] =='/':
            ends.append(k+1)
            
        # print(starts,ends)
        assert len(starts) == len(ends)
        new_line=line
        for st,end in zip(starts,ends):
          new_line=new_line[:st] + " "*(end-st+1)  + new_line[end+1:] 
        content[index]=new_line 
        continue
      if type_in_line==1:
        start_line.append(index)
      elif type_in_line==2:
        end_line.append(index)
    # assert len(start_line) == len(end_line),(start_line,end_line,content)
    for st,en in zip(start_line, end_line):
      content[st:en+1] = ["",] * (en-st+1)
    return content

class DatasetTesting:
    def __init__(self, pandas_file, tokenizer):
        """
            pandas_file: DataFrame object:
                +> source_code field
            tokenizer: tokenizer object huggingface
        """
        self.pandas_file =pandas_file.copy()
        colunms = self.pandas_file.columns
        assert all([i in colunms for i in ["source_code"]])
        self.token = tokenizer
    def __len__(self):return len(self.pandas_file)
    def __getitem__(self,index):
        src = self.pandas_file.iloc[index].source_code
        
        # Step 1: remove comment 
        source_code_lines =src.split("\n") 
        source_code_line_remove_cmt = remove_comment(source_code_lines)

        map_output  = {
            "source_code_lines_raw":source_code_lines,
            "source_code_line_remove_cmt":source_code_line_remove_cmt, #step1
        }
        # Step 2 tokenizer 
        c=0
        source_code_line_tokenizers = []
        input_ids = []
        mapping_line_to_line_in_source_raw= []
        mapping_token_to_line_source_tokenizer =[]
        for index,line in enumerate(map_output['source_code_line_remove_cmt']):
            if line.strip()=="":continue
            mapping_line_to_line_in_source_raw.append(index)
            line=line.strip()
            line  = ' '.join(line.split())
            source_code_line_tokenizers.append(line)
            token_line = self.token(line, add_special_tokens=False)['input_ids']
            if len(input_ids ) == 0:
                token_line=[self.token.bos_token_id,] + token_line 
            if len(input_ids) == len(map_output['source_code_line_remove_cmt'])-2:
                token_line=token_line+[self.token.eos_token_id,]  
            mapping_token_to_line_source_tokenizer.extend([c,] * len(token_line))
            c=c+1 
            input_ids.extend(token_line)
        
        map_output['input_ids'] = input_ids
        map_output['mapping_token_to_line_source_tokenizer']=mapping_token_to_line_source_tokenizer
        map_output['mapping_line_to_line_in_source_raw']=mapping_line_to_line_in_source_raw
        map_output['source_code_line_tokenizers']=source_code_line_tokenizers

        return map_output


class DatasetTraining:
    def __init__(self, pandas_file, tokenizer):
        """
            pandas_file: DataFrame object:
                +> source_code field
                +> line_error: List[int]: list[index_of_error_line] start with 0
                +> error_type: List[int]: list[index_of_class_error] start with 1
                +> len(line_error) == len(error_type)
            tokenizer: tokenizer object huggingface
        """
        self.pandas_file =pandas_file.copy()
        colunms = self.pandas_file.columns
        assert all([i in colunms for i in ["source_code","line_error","error_type"]])   
        self.token = tokenizer
    def __len__(self,):return len(self.pandas_file)
    def __getitem__(self, index):
        sample = self.pandas_file.iloc[index] 
        source_code = sample.source_code
        
        label_line_index = ast.literal_eval(str(sample.line_error))
        label_line_error=ast.literal_eval(str(sample.error_type))
        assert (len(label_line_index) == len(label_line_error)), f"Number of line errors must equal number of error types {len(label_line_error)} - {len(label_line_index)}"
        label_line_index=[i -1 for i in label_line_index if i!=-1]
        label_line_error = [i for i in label_line_error if i!=-1]
        source_code_lines =source_code.split("\n") 
        source_code_line_remove_cmt = remove_comment(source_code_lines)

        map_output  = {
            "source_code_lines_raw":source_code_lines,
            "source_code_line_remove_cmt":source_code_line_remove_cmt, #step1
        }
        # Step 2 tokenizer 
        c=0
        source_code_line_tokenizers = []
        input_ids = []
        mapping_line_to_line_in_source_raw= []
        mapping_token_to_line_source_tokenizer =[]
        label = []
        for index,line in enumerate(map_output['source_code_line_remove_cmt']):
            if line.strip()=="":continue
            mapping_line_to_line_in_source_raw.append(index)
            line=line.strip()
            line  = ' '.join(line.split())
            source_code_line_tokenizers.append(line)
            token_line = self.token(line, add_special_tokens=False)['input_ids']
            if len(input_ids ) == 0:
                token_line=[self.token.bos_token_id,] + token_line 
            if len(input_ids) == len(map_output['source_code_line_remove_cmt'])-2:
                token_line=token_line+[self.token.eos_token_id,]  
            mapping_token_to_line_source_tokenizer.extend([c,] * len(token_line))
            c=c+1 
            input_ids.extend(token_line)

            # Check index line is in 
            if index in label_line_index: # line nay co loi
                class_error = label_line_error[ label_line_index.index(index) ]
                label.append(class_error)
            else:
                label.append(0) 
        
        map_output['input_ids'] = input_ids
        map_output['mapping_token_to_line_source_tokenizer']=mapping_token_to_line_source_tokenizer
        map_output['mapping_line_to_line_in_source_raw']=mapping_line_to_line_in_source_raw
        map_output['source_code_line_tokenizers']=source_code_line_tokenizers   
        map_output['label_sentence_level'] = label
        return map_output

    