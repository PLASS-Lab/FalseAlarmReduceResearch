#
# Class which represents the config of the tool
#

from SecurityScanner import SecurityScanner
import configparser
import glob
import os.path
class AnalyzeToolConfig(object):
	def __init__(self, configFile):
		self.config = configparser.ConfigParser()
		self.config.read('config.cfg')
		self.repDirectory = self.config.get('General', 'repDirectory')
		self.ccpptestsuiteFolderPath = self.config.get('General', 'ccpptestsuiteFolderPath')
		self.ccpptestsuitePath = self.config.get('General', 'ccpptestsuitepath')
		self.ccpptestsuitepathFolder = self.config.get('General', 'ccpptestsuitepathFolder')
		self.scanners = self.config.get('General', 'scanners')
		self.tmpCppDataPath = self.config.get('General', 'tmpCCppData')
		self.ccppScanners = set()
		self.samateCCPPFilePath = self.config.get('General', 'samateCCPPFilePath')
		self.securityModelPath = self.config.get('General', 'securityModelPath')
		self.cweMappingsPath = self.config.get('General', 'cweMappingsPath')

		self.buildScannerList()
		self.allowedFileTypes = self.config.get('General', 'allowedFileTypes').split(',')
		
	#build scannerList for the scanners which are defined in the config
	def buildScannerList(self):
		if(len(self.scanners)<=0):
			print("no scanners defined. returning...")
			return;
		scannerNames = self.scanners.split(',')
		for sc in scannerNames:
			securityScanner = SecurityScanner(sc, self)
			
			if(securityScanner.type == 'ccpp'):
				self.ccppScanners.add(securityScanner)
			
	def getCCppScannerList(self):
		return self.ccppScanners;