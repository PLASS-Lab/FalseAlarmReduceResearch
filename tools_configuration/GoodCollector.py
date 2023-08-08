#! /usr/bin/env/python 3.1
#
# run through files and parse errors from the files which are indicated by comments
#

from AnalyzeToolConfig import AnalyzeToolConfig
from Issue import Issue
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring, parse
import glob
import os.path
import re
import sys
import os

class GoodCollector(object):
	def __init__(self, config):
		self.config = config

	#collect flaws from the file
	def collectGood(self,filePath):
		lineNumber = 1
		goodFunctionFound = False
		goodFunctionPattern = "good\d{1,5}\(\).*" #for goodFunction
		cweEntry = os.path.basename(filePath).split("_")[0]
		issueList = []
		isInComment = False;
		isBlank = False;
		newGood = False;
  
		for line in open(filePath):
			trimedLine = line.strip()
			commentStart = trimedLine.find('/*')
			commentEnd = trimedLine.find('*/')
   
			#use search regex to find the line match
			match = re.search(goodFunctionPattern, trimedLine)
			if(match != None and ((commentStart==-1 and commentEnd==-1) or not (match.start() >=commentStart and match.start()<=commentEnd))):
				correctMatch = True
			else:
				correctMatch = False
				
			if (goodFunctionPattern and correctMatch and not ";" in trimedLine and not isInComment):
				goodFunctionFound = True
				openBracketCount = 0
				if(not isBlank or newGood):
					issue = Issue(filePath, cweEntry, -1)
					issue.startLine = lineNumber
					newGood = False
			
			if(isBlank and (len(trimedLine)==0 or trimedLine.startswith('/*')) ):
				isBlank = True
			else:
				isBlank = False
				
			if(goodFunctionFound):
				if('{' in trimedLine and not trimedLine.startswith('/*') and not trimedLine.startswith('*')):
					openBracketCount = openBracketCount + 1
				if('}' in trimedLine and not trimedLine.startswith('/*') and not trimedLine.startswith('*')):
					openBracketCount = openBracketCount - 1		
					if(openBracketCount==0):
						issue.endLine = lineNumber
						issueList.append(issue)
						isBlank = True
						newGood = True
						goodFunctionFound = False

			lineNumber+=1
		return issueList
	
	#write the Issue
	def writeXML(self, flawMap, fileXML):
		for key, value in flawMap.items():
			for issue in value:
				elem = issue.getXMLElement(fileXML)
						
	def collect(self, testsuiteLanguage):
		if(testsuiteLanguage=='ccpp'):
			searchPath = self.config.ccpptestsuitePath
			searchPathFolder = self.config.ccpptestsuitepathFolder
			outputPath = self.config.tmpCppDataPath
		
		if (os.path.exists(outputPath+"existingGood.xml")):
			print("existingIssue file already exists. aborting...");
			# return
			
		print("searchPath="+searchPath)
		
		files = glob.glob(searchPath)
		filesInsideFolder = glob.glob(searchPathFolder)
		files = files + filesInsideFolder
	
		root = Element('files')
		
		for file in files:	
			#AW20130717 ensure only defined file extensions are processed
			if(any(file.endswith(x) for x in self.config.allowedFileTypes)):		
				goodList = self.collectGood(file)
				if(goodList!=None):
					lastStartLine = -1;
					lastEndLine = -1;
					writeIssueCnt = 0;
					for good in goodList:
						if(lastStartLine!=good.startLine and lastEndLine!=good.endLine or good.lineNumber!=-1):
							fileXML = SubElement(root, 'file', {'path' : file})
							good.getXMLElement(fileXML)
							lastStartLine = good.startLine;
							lastEndLine = good.endLine;
							writeIssueCnt+=1
		
		#write the result as xml
		with open(outputPath+'existingGood.xml', 'w') as f:
			f.write(parseString(tostring(root)).toprettyxml())
		
		#endTime = time.time()
		#print("flaws collected in "+str((endTime-startTime))+" seconds")

if __name__ == '__main__':
	testsuiteLanguage = 'java'
	if(len(sys.argv)>1):
		testsuiteLanguage = sys.argv[1]
	config = AnalyzeToolConfig('config.cfg')
	
	collector = GoodCollector(config)
	collector.collect(testsuiteLanguage)

