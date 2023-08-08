#
# Simple class for encapsulating different kinds of security scanners
#
import time
class SecurityScanner(object):
	def __init__(self, name, cfg):
		self.name = name
		self.cfg = cfg
		
		if(cfg.config.has_option(name, 'options')):
			self.options = cfg.config.get(name, 'options')
		
		if(cfg.config.has_option(name, 'executable')):
			self.executable = cfg.config.get(name, 'executable')
			
		if(cfg.config.has_option(name, 'afterFileOptions')):
			self.afterFileOptions = cfg.config.get(name, 'afterFileOptions')
		
		if(cfg.config.has_option(name, 'wrapper')):
			self.wrapper = cfg.config.get(name, 'wrapper')
		
		if(cfg.config.has_option(name, 'type')):
			self.type = cfg.config.get(name, 'type')
			
			
		if(cfg.config.has_option(name, 'scanFolder')):
			self.scanFolder = cfg.config.get(name, 'scanFolder')
		else:
			self.scanFolder = False
			
		if(cfg.config.has_option(name, 'outputFileCsv')):
			self.outputFileCsv = cfg.config.get('General', 'repDirectory') + cfg.config.get(name, 'outputFileCsv')	

		if(cfg.config.has_option(name, 'outputFileTxt')):
			self.outputFileTxt = cfg.config.get('General', 'repDirectory') + cfg.config.get(name, 'outputFileTxt')
		
	def getName(self):
		return self.name

	def getOutputFileCsv(self):
		return self.outputFileCsv

	def getOutputFileTxt(self):
		return self.outputFileTxt
 
	# return the cmd String which can be run from the commandline
	def getCmdString(self, file, fileName):
		commandList = []
		commandList.append(self.executable)
		
		if(hasattr(self, 'options')):
			commandList.append(self.options)
		
		commandList.append(file)
		
		if(hasattr(self, 'afterFileOptions')):
			commandList.append(self.afterFileOptions)
		if(self.name == "infer"):
			# quick
			cmd = " ".join(commandList[0:2]) + commandList[2] + " " +  commandList[3]
		else:
			cmd = " ".join(commandList)

		cmd = cmd.replace("#filename", fileName)
		return cmd