#! /usr/bin/env/python 3.1
#
# handle the transformation of the scanner results files. 
# use MOT as transformattor
# author: Andreas Wagner
#
from AnalyzeToolConfig import AnalyzeToolConfig
import os.path
import glob
import py_common
from csv import reader

class ReportLine(object):
    def __init__(self, file, line, message):
        self.file = file
        self.line = line
        self.message = message
        

class TransformTool(object):
    def __init__(self, config):
        self.config = config

    def transformResult(self, scannerName, cwefileList, tmpDataDir):
        summary_file = os.getcwd() + "/tmpData/" + scannerName + "_report.csv"
        with open(summary_file, "w") as outfile:
            outfile.write('tool,file,line,value,detail\n')
            for cwefile in cwefileList:
                cwe_name = cwefile.split("/")[3]
                files = glob.glob(tmpDataDir+ "/*" + cwe_name + "*.csv")
                for file in files:
                    with open(file, 'r') as read_obj:
                        csv_reader = reader(read_obj)
                        header = next(csv_reader)
                        if header != None:
                            for row in csv_reader:
                                file = str(row[0]).strip()
                                if("framac" in scannerName or "infer" in scannerName or "pvsstudio"):
                                    file_full_path = str(row[0]).strip().split('/')
                                    file = file_full_path.pop()
                                line = str(row[1]).strip()
                                value = str(row[2]).strip()
                                write_line = scannerName + ',' + file + "," + line + "," + value + "\n"
                                outfile.write(write_line)  
                            
    def transformResultCWE(self, cwefileList, ccppScannerList):
        list_report = dict()
        
        if(len(ccppScannerList)>0):
            for scanner in ccppScannerList:
                scannerName = scanner.getName()
                result_path_tools =  os.getcwd() + "/tmpData/" + scannerName + "_report.csv"
                with open(result_path_tools, 'r') as f:
                        lines = []
                        for line in f.readlines(): 
                            content = line.split(",")
                            if content[0] == 'Tool':continue;
                            report_by_line = ReportLine(content[1], content[2], content[3])
                            lines.append(report_by_line)
                        list_report[scannerName] = lines
        print("======== start convert to CWE ========")
        for cwefile in cwefileList:
            cwe_name = cwefile.split("/")[3]
            summary_file = os.getcwd() + "/tmpData/" + cwe_name + "_report.csv"
            print(cwe_name)
            with open(summary_file, "w") as outfile:
                outfile.write('file,line,tool,message\n')      
                cwe_name = cwefile.split("/")[3]
                file_list_by_cwe = sorted(glob.glob(cwefile+ "*", recursive=True))
                lastFolder = glob.glob(cwefile + "/s0*")
                if(len(lastFolder) != 0):
                    file_list_by_cwe = list()
                    for subfile in lastFolder:
                        sub_testcaseFile = glob.glob(subfile + "/CWE*")
                        file_list_by_cwe += sub_testcaseFile
                
                for file in file_list_by_cwe:
                    file_name = os.path.basename(file)
                    for scanner in ccppScannerList:
                        scannerName = scanner.getName()
                        reports = list_report[scannerName]
                        for item in reports:
                            file_by_tool = item.file
                            if file_by_tool == file_name and file_by_tool.startswith("CWE"): #s1
                                write_line = file_by_tool + ',' + item.line + "," + scannerName + "," + item.message
                                outfile.write(write_line)
        print("======== end convert to CWE ========")
        
    def transformResults(self):
        cfg = self.config
        ccppScannerList = cfg.getCCppScannerList()
        searchPathCCpp = cfg.ccpptestsuiteFolderPath
        cwefileList = glob.glob(searchPathCCpp)
        if(len(ccppScannerList)>0):
            for scanner in ccppScannerList:
                scannerName = scanner.getName()
                result_path = cfg.tmpCppDataPath + scannerName
                self.transformResult(scannerName, cwefileList, result_path)
                
        self.transformResultCWE(cwefileList, ccppScannerList)
        
if __name__ == '__main__':
    cfg = AnalyzeToolConfig('config.cfg')
    
    tool = TransformTool(cfg)
    tool.transformResults()
