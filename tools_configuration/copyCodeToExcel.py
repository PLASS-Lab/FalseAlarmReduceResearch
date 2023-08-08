from cProfile import label
from IssueComparison import IssueComparison
from ComparisonResultHolder import ComparisonResultHolder
from Issue import Issue
from AnalyzeToolConfig import AnalyzeToolConfig
from collections import OrderedDict
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
import os
from pylab import *
from ScannerCWEMapping import ScannerCWEMapping
import csv
from ScannerIssueHolder import ScannerIssueHolder
from SecurityModel import SecurityModel
from SecurityModelComparision import SecurityModelComparision
from HTMLReport import HTMLReport
import sys
import random
import numpy as np
import re

class Issues(object):
  def __init__(self, code, label, file, error_line, start_line = 0, end_line = 0):
    self.file = file 
    self.code = code
    self.label = label
    self.error_line = error_line
    self.start_line = start_line
    self.end_line = end_line
    
def readManifestFile(root):
		flawMap = dict()
		for testcase in root:
			issueList = []
			for file in testcase.iter("file"):
				fileName = os.path.basename(file.get("path")).lower()
				if("w32" in fileName or "wchar" in fileName):
								continue
				if(file.findall("flaw")):
					for item in file.findall("flaw"):
						newIssue = Issue(fileName,item.get("name").split(":")[0], item.get("line"))
						issueList.append(newIssue)

				issueComparision = IssueComparison(fileName)
				issueComparision.addExistingIssues(issueList)
				flawMap[fileName] = issueComparision
		return flawMap
    
def remove_comments(string):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
    def _replacer(match):
        if match.group(2) is not None:
            return "" 
        else: 
            return match.group(1) 
    return regex.sub(_replacer, string)
      
def removeComments(string):
  string = re.sub(re.compile("/\*.*?\*/",re.DOTALL ) ,"" ,string) # remove all occurrences streamed comments (/*COMMENT */) from string
  string = re.sub(re.compile("//.*?\n" ) ,"" ,string) # remove all occurrence single-line comments (//COMMENT\n ) from string
  return string 
if __name__ == '__main__':
  
  eTree_good = ET.parse('./existingGood.xml')
  eTree_issues = ET.parse('./existingIssues.xml')
  path_result = 'CWE758.csv'
  
  root_good = eTree_good.getroot()
  root_isssues = eTree_issues.getroot()
  
  f = open(path_result, 'w')
  writer = csv.writer(f)
  writer.writerow(["file", "code", "label", "error_line", "start_line", "end_line"])
  
  existingIssuesFile ='manifest.xml'
  eTree = ET.parse(existingIssuesFile)
  root = eTree.getroot()
  flawMap = readManifestFile(root)

  list_line_good = []
  for files in root_good:
    for file in files.iter("file"):
      file
      fileName = os.path.basename(file.get("path"))
      if("w32" in fileName or "wchar" in fileName):
        continue
      # if("CWE758" in fileName):
      if(file.findall("issue")):
        for item in file.findall("issue"):
          path = file.get("path")
          with open(path, "r") as code:
            lines = code.readlines()
            count = 0
            list_lines = ""
            for line in lines:
              count +=1
              if count >= int(item.get("startLine")) + 2 and count <= int(item.get("endLine")) - 2:
                list_lines += removeComments(remove_comments(line))
            issues = Issues(fileName, list_lines, 0, 0)
            list_line_good.append(issues)
  
  list_line_issues = []            
  for files in root_isssues:
    for file in files.iter("file"):
      fileName = os.path.basename(file.get("path"))
      if("w32" in fileName or "wchar" in fileName):
        continue
      # if("CWE758" in fileName):
      if(file.findall("issue")):
        for item in file.findall("issue"):
          path = file.get("path")
          # if(fileName == 'CWE190_Integer_Overflow__short_rand_multiply_83a.cpp'):
          #   print("hello")
          flaw = flawMap[fileName.lower()]
          issue = flaw.existingIssues[0]
          file_path_from_issue = issue.filePath
          # if(file_path_from_issue != fileName.lower()):
          #   path_replace = path.replace(fileName, file_path_from_issue)
          #   path = path_replace
          lineNumber = issue.lineNumber
          startLine = int(item.get("startLine"))  
          line_b = int(lineNumber) - int(startLine)
          line_b = 5 if line_b <= 0 else line_b 
          with open(path, "r") as code:
            lines = code.readlines()
            count = 0
            list_lines = ""
            for line in lines:
              count +=1
              startLine = int(item.get("startLine")) 
              endLine = item.get("endLine") 
              endLine = int(endLine) if endLine else startLine + 20 
              if count >= startLine + 2 and count <= endLine - 2:
                list_lines += removeComments(remove_comments(line))
              if not (line_b >= startLine and line_b <= endLine):
                line_b = random.randrange(1, endLine - startLine)
            issues = Issues(fileName, list_lines, 1, line_b, startLine, endLine)
            list_line_issues.append(issues)      
              
  summary_line = list_line_good + list_line_issues      
  np.random.shuffle(summary_line)
  for item in summary_line:
    writer.writerow([item.file, item.code, item.label, item.error_line, item.start_line, item.end_line])       
  f.close()  
      



    