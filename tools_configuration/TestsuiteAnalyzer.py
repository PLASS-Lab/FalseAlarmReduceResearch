#! /usr/bin/env/python 3.1
#
# analyze tool inspired from juliet-test-suite 'run_analysis_example_tool
#

from logging import warning
from AnalyzeToolConfig import AnalyzeToolConfig
import glob
import os
import py_common
import sys
import time
import dirutils
import json
import xml.etree.ElementTree as ET

# add parent directory to search path so we can use py_common
sys.path.append("..")

class TestsuiteAnalyzer(object):
	def __init__(self, config):
		self.config = config
		
	#run each security scanner for the file
	def run_example_tool(self, bat_file, scannerList):
			
		for sc in scannerList:
			#AW20130309 set use shell to true here, otherwise pipes don't work
			py_common.run_commands([sc.getCmdString(bat_file)], True)
			#print(sc.getCmdString(bat_file))
			
	# copied from pycommons and modified slightly to fit the needs
	def run_analysis(self,glob_needle, run_analysis_fx, scannerList):
		"""
		Helper method to run an analysis using a tool.  Takes a glob string to search
		for and a function pointer.
		"""
		files = glob.glob(glob_needle)
		lastDir = 'none'

		with open('something.txt', 'w+') as f:
			f.write(str(files))
		
		# run all the files using the function pointer
		for file in files:
			#AW20130717 ensure only defined file extensions are processed
			#! cần kiểm tra file được cho phép
			if(True):	
			# if(any(file.endswith(x) for x in self.config.allowedFileTypes)):	
				sys.stdout = open('./folder.csv', "a")
				print(file)
				sys.stdout = sys.__stdout__


				# change into directory with the file
				dir = os.path.dirname(file)
				parentPath = os.getcwd()

				# cd path to vulnerability folder
				# os.chdir(dir) #! 추가

				# run the the file
				# file = os.path.basename(file) #! 추가
				dirName = os.path.basename(dir)
				# if(dirName != 'CWE476_NULL_Pointer_Dereference'):
				# 	continue;
				lastFolder = glob.glob(file + "/s0*")
				if(len(lastFolder) != 0):
					for subfile in lastFolder:
							cweFolder = os.path.dirname(subfile)
							cweName = os.path.basename(cweFolder)
							dir = subfile
							dirName = os.path.basename(dir)
							fileName = cweName + "_" + dirName
							self.run_analyzer(parentPath, dir, fileName, scannerList)
				else:
					self.run_analyzer(parentPath, dir, dirName, scannerList)
        
	def run_analyzer(self, parentPath, dir, dirName, scannerList):
			"""
				run every tool with scanned files
			"""
			for sc in scannerList:
				path_csv = sc.getOutputFileCsv().replace("#filename", dirName)
				path_txt = sc.getOutputFileTxt().replace("#filename", dirName)
				dirutils.file_line_error_header(path_csv)
    
				# testcases for infer tools
				if(sc.getName() == 'infer'):
						result_path = sc.getOutputFileCsv().replace("#filename", dirName).replace(".csv", "")
						py_common.run_commands(["cd " + dir + " && make clean"], True)
						cmd = sc.getCmdString(dir,dirName)
						(output, err, exit, time) = dirutils.system_call(cmd, ".") 
						dirutils.tool_exec_log(path_txt, cmd, output, err, exit)
			
						sys.stdout = open(path_csv, "a")
						report_file = os.path.join(result_path, "report.json")
						if (os.path.exists(report_file)):
								with open(report_file) as json_report_file:
										data = json.load(json_report_file)
								if(data and len(data) != 0):
									with open(parentPath + '/infer_result.csv',"w") as f:
										f.write(str(dirName))
								for d in data:
										print(d['file'].strip(), ",", str(d['line']), ",", "\"" + d['qualifier'] + "\"")
						sys.stdout = sys.__stdout__
      
				elif(sc.getName() == 'clang'):

					py_common.run_commands(["cp ./clang.sh " + dir], True)
					print(os.getcwd())
					os.chdir('/home/huong/projects/VDCT/' + dir)
					cmd = "./clang.sh " + dirName
					(output, err, exit, time) = dirutils.system_call(cmd, ".") 
					dirutils.tool_exec_log(path_txt, cmd, output, err, exit)
				
					sys.stdout = open(path_csv, "a")
					print(err, file=sys.__stdout__)
					lines = err.splitlines()
					for line in lines:
							parsed = line.decode("utf-8").strip().split(":")
							if (len(parsed) >= 4 and not parsed[3].endswith('note')):
									print(os.path.basename(parsed[0]), ",", parsed[1], ",", parsed[3] + ":" + parsed[4])
					sys.stdout = sys.__stdout__
     
				elif(sc.getName() == 'cppcheck'):
						xml_report_path =  sc.getOutputFileCsv().replace("#filename", dirName).replace("csv", "xml")
						cmd = sc.getCmdString(dir,dirName)
						(output, err, exit, time) = dirutils.system_call(cmd, ".")
						dirutils.tool_exec_log(path_txt, cmd, output, err, exit)
						tree = ET.parse(xml_report_path)
						root = tree.getroot()
						errors = root[1]
						cppcheck_scanner_result = parentPath + '/tmpData/cppcheck-scanner.csv'
      
						with open(cppcheck_scanner_result,"a") as f:
							f.write(str(dirName) + ',' + str(len(errors)) + "\n")
						sys.stdout = open(path_csv, "a")
						for error in errors:
								error_id= error.attrib['id']
								cwe = error.attrib['cwe']
								for location in error:
										if (location.tag == "location"):
												print(os.path.basename(location.attrib['file']) + ",", location.attrib['line'] + ",", error_id + ",", cwe)
						sys.stdout = sys.__stdout__
				elif(sc.getName() == 'flawfinder'): #! need to check again
						cmd = sc.getCmdString(dir, '')
						(output, err, exit, time) = dirutils.system_call(cmd, ".") 
						dirutils.tool_exec_log(path_txt, cmd, output, err, exit)
						all_lines = output.splitlines()
						lines = []
						line_codes = []
						collect_flag = False
						for line in all_lines:
								dec = line.decode("utf-8").strip()
								if (collect_flag):
										lines.append(dec)
										if (len(dec.split(":")) >= 3):
												line_codes.append(True)
										else:
												line_codes.append(False)
								if dec == "FINAL RESULTS:":
										collect_flag = True
								if dec == "ANALYSIS SUMMARY:":
										break

						sys.stdout = open(path_csv, "a")
						for i in range(0,len(lines)):
								if (line_codes[i]):
										a = lines[i].split(":")
										filename = os.path.basename(a[0])
										line_no = a[1]
										error_message = ""
										j = 2
										while (j < len(a)):
												error_message = error_message + ":" + a[j]
												j = j + 1
										j = i + 1
										while (j < len(lines)):
												if (not line_codes[j]):
														error_message = error_message + " " + lines[j].strip()
														j = j + 1
												else:
														break;
										template_message = error_message.split()
										for item in template_message:
											if item.find("CWE") != -1:
												start = item.find('(')
												end = item.find(')')
												cwe = item[start + 1: end]
										print(filename, ",", line_no, ",", error_message , ",", cwe)
						sys.stdout = sys.__stdout__       
				elif(sc.getName() == 'framac'):
				
						# only run with *c file
						source_files = glob.glob(dir + "/*.c")
						for source in source_files:
							filename = os.path.basename(source)
							framac = 'frama-c -quiet -main ' + filename.split(".")[0] + "_bad " + source + " -cpp-extra-args='-I./1v3/juliet_suite-c-cplus/src/testcasesupport  -DINCLUDEMAIN -U__cplusplus'"
							(output, err, exit, time) = dirutils.system_call(framac, '.')
							print("========================== OUPUT", output)
							dirutils.tool_exec_log(path_txt, framac, output, err, exit)
							sys.stdout = open(path_csv, "a")
							lines = output.splitlines()
							i = 0
							while i < len(lines):
									line = lines[i].decode("utf-8")
									if (line[0] == '['):
											j = line.find("]");
											if (j != -1):
													parsed = line[j+1:].split(':')
													if (len(parsed) >= 3):
															fname = parsed[0].strip()
															line_no = parsed[1].strip()
															message = parsed[2].strip()
															if (i + 1 < len(lines)):
																	message = message + ":" + lines[i+1].decode("utf-8")
															if (fname != "main.c" and line_no.isdigit()):
																	print(fname + "," + line_no + "," + message)
									i = i + 1
							sys.stdout = sys.__stdout__
				elif(sc.getName() == 'pvsstudio'):
					
						path_csv = sc.getOutputFileCsv().replace("#filename", dirName)
						path_txt = sc.getOutputFileTxt().replace("#filename", dirName)
			
						print(os.getcwd())
						dirutils.file_line_error_header(path_csv)
      
						py_common.run_commands(["cp ./pvs_studio.sh " + dir], True)
						py_common.run_commands(["cp ./PVS-Studio.cfg " + dir], True)
      
						os.chdir('/home/huong/projects/VDCT/' + dir)
						cmd = "./pvs_studio.sh " + dirName
						(output, err, exit, time) = dirutils.system_call(cmd, ".") 
						dirutils.tool_exec_log(path_txt, cmd, output, err, exit)
						log_file = glob.glob("/home/huong/projects/VDCT/tmpData/CCPP/pvsstudio/" + dirName + "/*.json")
						sys.stdout = open(path_csv, "a")
						for log in log_file:
							if (os.path.exists(log)):
									with open(log, "r") as json_report_file:
										data = json.load(json_report_file)
									if len(data['warnings']) == 0:
										continue
									warnings = data['warnings'][0]
									# code = warnings['code']
									cwe = warnings['cwe']
									message = warnings['message']
									positions = warnings['positions'][0]
									file = positions['file']
									line = positions['line']
									print(file, ",", line, ",", message, ",", cwe), 
						sys.stdout = sys.__stdout__
						lastDir=dir
				else:
					if(not sc.scanFolder):
						py_common.run_commands([sc.getCmdString(file,file)], True)
					elif(sc.scanFolder and lastDir!=dir):
						py_common.run_commands([sc.getCmdString(dir,dirName)], True)
						lastDir=dir
      
				os.chdir(sys.path[0])
		
	def runAnalyze(self):
		cfg = self.config
		ccppScannerList = cfg.getCCppScannerList()
		searchPathCCpp = cfg.ccpptestsuiteFolderPath #+ "\\testcases\\*\\CWE126_Buffer_Overread__CWE129_large_01.c"
		startAnalysis = time.time()
		
		if(len(ccppScannerList)>0):
			print("start C/Cpp testcases")
			ccppStartAnalysis = time.time()
			self.run_analysis(searchPathCCpp, self.run_example_tool, ccppScannerList)
			ccppEndAnalysis = time.time()
			ccppOverallTime = (ccppEndAnalysis - ccppStartAnalysis)
		
		endAnalysis = time.time()
		overallTime = (endAnalysis - startAnalysis)
		print("Overall analysis took " + str(overallTime)+" seconds;") 
		
		if(len(ccppScannerList)>0):
			print("\t C/C++="+str(ccppOverallTime)+" seconds")
			
if __name__ == '__main__':
	cfg = AnalyzeToolConfig('config.cfg')
	
	tool = TestsuiteAnalyzer(cfg)
	tool.runAnalyze()



