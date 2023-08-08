#! /usr/bin/env/python 3.1
#
# tool which converts the output of several security checkers to an own format which 
# is needed for further processing
#
from CppcheckResultConverter import CppcheckResultConverter
from AnalyzeToolConfig import AnalyzeToolConfig
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString

def convertResult(scanner, tmpDataDir):
	outputFile = config.config.get(scanner, 'outputFile')
	
	if(scanner=='cppcheck'):
		converter = CppcheckResultConverter(outputFile, tmpDataDir)
	root = converter.getXML()
	with open(tmpDataDir+scanner+'_converted_file.xml', 'w') as f:
		f.write(parseString(ET.tostring(root)).toxml())

if __name__ == '__main__':
	config = AnalyzeToolConfig('config.cfg')
	
	converterMap = dict()
	
	for scanner in config.getCCppScannerList():
		print("convert file for "+scanner.name)
		convertResult(scanner.name, config.tmpCppDataPath)
	
	for scanner in config.getJavaScannerList():
		convertResult(scanner.name, config.tmpJavaDataPath)	