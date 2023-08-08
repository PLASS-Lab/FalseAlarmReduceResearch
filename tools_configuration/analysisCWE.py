import glob
import sys

testcaseFile = sorted(glob.glob('./Juliet-Suite-C_Cplus-Dataset/CWE*', recursive=True))
path_csv = './CWE_summary.csv';
sys.stdout = open(path_csv, "a")
for folder in testcaseFile:
  cwe_name =folder.split('/')[2]
  lastFolder = glob.glob(folder + "/s0*")
  count = 0
  if(len(lastFolder) != 0):
    for subfile in lastFolder:
      sub_testcaseFile = glob.glob(subfile + "/CWE*")
      count = count + len(sub_testcaseFile)
    print(cwe_name,",",count)
    continue;
  file_count = glob.glob(folder + "/CWE*")   
  print(cwe_name,",",len(file_count))
sys.stdout = sys.__stdout__
      
							