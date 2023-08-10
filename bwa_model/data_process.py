import os,sys,json,glob
import pandas as pd
import logging
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import argparse
from xmljson import badgerfish as bf
from xml.etree.ElementTree import fromstring
from json import dumps
from tqdm import tqdm 
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)
current_path = os.getcwd()
root_zip = current_path + '/output'
dataset_path = 'julietsuite'
torch.cuda.empty_cache()


classes=[
        'nan',
        'CWE-114: Process Control',
       'CWE-121: Stack-based Buffer Overflow',
       'CWE-135: Incorrect Calculation of Multi-Byte String Length',
       'CWE-126: Buffer Over-read', 
       'CWE-122: Heap-based Buffer Overflow',
       'CWE-123: Write-what-where Condition',
       "CWE-124: Buffer Underwrite ('Buffer Underflow')",
       'CWE-127: Buffer Under-read',
       'CWE-134: Uncontrolled Format String',
       'CWE-015: External Control of System or Configuration Setting',
       'CWE-176: Improper Handling of Unicode Encoding',
       'CWE-188: Reliance on Data/Memory Layout',
       'CWE-190: Integer Overflow or Wraparound',
       'CWE-191: Integer Underflow (Wrap or Wraparound)',
       'CWE-194: Unexpected Sign Extension',
       'CWE-195: Signed to Unsigned Conversion Error',
       'CWE-196: Unsigned to Signed Conversion Error',
       'CWE-197: Numeric Truncation Error',
       'CWE-222: Truncation of Security-relevant Information',
       'CWE-223: Omission of Security-relevant Information',
       'CWE-226: Sensitive Information Uncleared Before Release',
       'CWE-023: Relative Path Traversal',
       'CWE-242: Use of Inherently Dangerous Function',
       "CWE-244: Improper Clearing of Heap Memory Before Release ('Heap Inspection')",
       'CWE-247: DEPRECATED (Duplicate): Reliance on DNS Lookups in a Security Decision',
       'CWE-252: Unchecked Return Value',
       'CWE-253: Incorrect Check of Function Return Value',
       'CWE-256: Plaintext Storage of a Password',
       'CWE-259: Use of Hard-coded Password',
       'CWE-272: Least Privilege Violation',
       'CWE-273: Improper Check for Dropped Privileges',
       'CWE-284: Improper Access Control',
       'CWE-319: Cleartext Transmission of Sensitive Information',
       'CWE-321: Use of Hard-coded Cryptographic Key',
       'CWE-325: Missing Required Cryptographic Step',
       'CWE-327: Use of a Broken or Risky Cryptographic Algorithm',
       'CWE-328: Reversible One-Way Hash',
       'CWE-338: Use of Cryptographically Weak Pseudo-Random Number Generator (PRNG)',
       'CWE-364: Signal Handler Race Condition',
       'CWE-366: Race Condition within a Thread',
       'CWE-367: Time-of-check Time-of-use (TOCTOU) Race Condition',
       'CWE-369: Divide By Zero', 'CWE-036: Absolute Path Traversal',
       'CWE-377: Insecure Temporary File',
       'CWE-390: Detection of Error Condition Without Action',
       'CWE-391: Unchecked Error Condition',
       'CWE-396: Declaration of Catch for Generic Exception',
       'CWE-397: Declaration of Throws for Generic Exception',
       'CWE-398: Indicator of Poor Code Quality',
       "CWE-400: Uncontrolled Resource Consumption ('Resource Exhaustion')",
       "CWE-401: Improper Release of Memory Before Removing Last Reference ('Memory Leak')",
       'CWE-404: Improper Resource Shutdown or Release',
       'CWE-415: Double Free', 'CWE-416: Use After Free',
       'CWE-426: Untrusted Search Path',
       'CWE-427: Uncontrolled Search Path Element',
       'CWE-440: Expected Behavior Violation',
       'CWE-457: Use of Uninitialized Variable',
       'CWE-459: Incomplete Cleanup',
       'CWE-464: Addition of Data Structure Sentinel',
       'CWE-467: Use of sizeof() on a Pointer Type',
       'CWE-468: Incorrect Pointer Scaling',
       'CWE-469: Use of Pointer Subtraction to Determine Size',
       'CWE-475: Undefined Behavior for Input to API',
       'CWE-476: NULL Pointer Dereference',
       'CWE-478: Missing Default Case in Switch Statement',
       'CWE-479: Signal Handler Use of a Non-reentrant Function',
       'CWE-480: Use of Incorrect Operator',
       'CWE-481: Assigning instead of Comparing',
       'CWE-482: Comparing instead of Assigning',
       'CWE-483: Incorrect Block Delimitation',
       'CWE-484: Omitted Break Statement in Switch',
       'CWE-500: Public Static Field Not Marked Final',
       'CWE-506: Embedded Malicious Code', 'CWE-510: Trapdoor',
       'CWE-511: Logic/Time Bomb',
       'CWE-526: Information Exposure Through Environmental Variables',
       'CWE-534: Information Exposure Through Debug Log Files',
       'CWE-535: Information Exposure Through Shell Error Message',
       'CWE-546: Suspicious Comment', 'CWE-561: Dead Code',
       'CWE-562: Return of Stack Variable Address',
       "CWE-563: Assignment to Variable without Use ('Unused Variable')",
       'CWE-570: Expression is Always False',
       'CWE-571: Expression is Always True',
       'CWE-587: Assignment of a Fixed Address to a Pointer',
       'CWE-588: Attempt to Access Child of a Non-structure Pointer',
       'CWE-590: Free of Memory not on the Heap',
       'CWE-591: Sensitive Data Storage in Improperly Locked Memory',
       'CWE-605: Multiple Binds to the Same Port',
       'CWE-606: Unchecked Input for Loop Condition',
       'CWE-615: Information Exposure Through Comments',
       'CWE-617: Reachable Assertion',
       'CWE-620: Unverified Password Change',
       'CWE-665: Improper Initialization',
       'CWE-666: Operation on Resource in Wrong Phase of Lifetime',
       'CWE-667: Improper Locking',
       'CWE-672: Operation on a Resource after Expiration or Release',
       'CWE-674: Uncontrolled Recursion',
       'CWE-675: Duplicate Operations on Resource',
       'CWE-676: Use of Potentially Dangerous Function',
       'CWE-680: Integer Overflow to Buffer Overflow',
       'CWE-681: Incorrect Conversion between Numeric Types',
       'CWE-685: Function Call With Incorrect Number of Arguments',
       'CWE-688: Function Call With Incorrect Variable or Reference as Argument',
       'CWE-690: Null Deref from Return',
       'CWE-758: Reliance on Undefined, Unspecified, or Implementation-Defined Behavior',
       'CWE-761: Free of Pointer not at Start of Buffer',
       'CWE-762: Mismatched Memory Management Routines',
       'CWE-773: Missing Reference to Active File Descriptor or Handle',
       'CWE-775: Missing Release of File Descriptor or Handle after Effective Lifetime',
       'CWE-780: Use of RSA Algorithm without OAEP',
       'CWE-785: Use of Path Manipulation Function without Maximum-sized Buffer',
       'CWE-789: Uncontrolled Memory Allocation',
       "CWE-078: Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')",
       'CWE-832: Unlock of a Resource that is not Locked',
       "CWE-835: Loop with Unreachable Exit Condition ('Infinite Loop')",
       "CWE-843: Access of Resource Using Incompatible Type ('Type Confusion')",
       "CWE-090: Improper Neutralization of Special Elements used in an LDAP Query ('LDAP Injection')"
    ]
def pre_data(): 
  # read manifest.xml to collect CWE testcases file
  data=bf.data(fromstring(open(f"{dataset_path}/manifest.xml","r").read()))
  data=data['container']
  data=data['testcase']
  data[0]
  ds=[]

  for i in data:
    i=i['file'] 
    if isinstance(i,(list,tuple)):
      ds.extend(i)
    else: ds.append(i)
  ds_pd=[]

  for i in ds:
    path = i['@path'] 
    flaw =i.get("flaw",{"@line":None,"@name":None})
    if isinstance(flaw, (list, tuple)):
      for fl in flaw:
        line = fl['@line']
        error_type = fl['@name']
        ds_pd.append({
            "File":path,
            "Line":line,
            "CWE":error_type
        })
    else:
      line = flaw['@line']
      error_type = flaw['@name']
      ds_pd.append({
          "File":path,
          "Line":line,
          "CWE":error_type
      })
  data_final=pd.DataFrame(ds_pd) 
  
  # mapping path to CWE testcases file
  all_file=list(glob.glob(current_path + "/" + dataset_path + "/*/*")) + list(glob.glob(current_path + "/" + dataset_path + "/*/*/*"))
  all_file=list(set(all_file))
  all_file={os.path.basename(i):i for i in all_file }
  
  # filter path is null of testcases file
  data_final['path']=data_final['File'].map(all_file)
  data_final[data_final.path.isna()].shape 
  data_final=data_final[~data_final.path.isna()]
  data_final['Error'] = data_final['CWE'].apply(get_cwe)
  data_final['content'] = data_final['path'].apply(lambda x:open(x,"r").read())
  data_final['error_line'] = data_final[['content','Line']].apply(get_line_vul, axis=1)
  return data_final
  
def rx(x):
  a=[]
  for i in x:
    if str(i)=='nan':
      a.append(10000000000)
    else:
      a.append(int(i))
  return a

def split_data(data_final):
  skf = StratifiedKFold(n_splits=2)
  data_final=data_final.reset_index()
  map_label={i:k for k,i in enumerate(data_final.CWE.unique())}
  phase_2=None
  phase_1=None
  for train_index, test_index in skf.split(data_final.index, data_final.CWE.map(map_label)):
    # print(train_index,test_index)
    phase_2=test_index
    phase_1=train_index


  skf2 = StratifiedKFold(n_splits=5)
  data_phase_2=data_final.loc[phase_2].copy().reset_index()
  data_pharse_1 = data_final.loc[phase_1].copy().reset_index()
  train_pharse_1=None
  val_pharse_1=None
  for train_index, test_index in skf2.split(data_pharse_1.index, data_pharse_1.CWE.map(map_label)):
    train_pharse_1=train_index
    val_pharse_1=test_index

  train_pharse1=data_pharse_1.loc[train_pharse_1].copy()
  val_pharse1=data_pharse_1.loc[val_pharse_1].copy()
  return train_pharse1, val_pharse1, data_phase_2

def save_data(data, type):
  source_code=data.groupby("File")['content'].first().reset_index(name="source_code")
  line_error=data.groupby("File")['Line'].apply(rx).reset_index(name="line_error")
  error_type=data.groupby("File")['Error'].apply(list).reset_index(name="error_type")
  ds = pd.merge(source_code, line_error, on='File', how='left')
  ds = pd.merge(ds, error_type, on='File', how='left')  
  save_data= ds
  if type == "train":
    save_data.to_csv("dataset/train.csv",index=False)
  elif type == "val":
    save_data.to_csv("dataset/val.csv",index=False)
  elif type == "test":
    save_data.to_csv("dataset/test.csv",index=False)
  
  
def get_line_vul(sample):
  content=sample[0]
  if np.isnan(sample[1]):
    return ""
  line=int(sample[1]) 
  return content.split("\n")[line-1]

def get_cwe(cwe):
  if cwe == None:
    return 0
  else:
    return classes.index(cwe)

    
def main():
    parser = argparse.ArgumentParser()
    env_args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_args.n_gpu = torch.cuda.device_count()
    env_args.device = device
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S', filename='train.log', level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, env_args.n_gpu)
    
    data_final = pre_data()
    train_pharse1, val_pharse1, data_phase_2 = split_data(data_final)
    save_data(train_pharse1, "train")
    save_data(val_pharse1, "val")
    save_data(data_phase_2, "test")
    
if __name__ == "__main__":
    main()