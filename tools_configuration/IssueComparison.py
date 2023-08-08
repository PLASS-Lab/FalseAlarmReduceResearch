#
# class which is used for comparing the exisiting issues (documented in the juliet testsuite)
# with the ones which are found by the security scanners
#

from IssueComparisionResult import IssueComparisonResult
from ScannerIssueHolder import ScannerIssueHolder
class IssueComparison(object):
	
	def __init__(self, fileName):
		self.existingIssues = []
		self.foundIssues = dict()
		self.correctMatchCnt = 0
		self.fileName = fileName
		self.noneMatching = 0
		
	# add an issue found by the flaw collector (documented in the juliet testsuite)
	def addExistingIssues(self, existingIssues):
		self.existingIssues = existingIssues
		
		issueList = []
		for issue in existingIssues:
			cwe = issue.category.replace('CWE', '')
			existingScannerIssue = ScannerIssueHolder(issue.filePath, cwe, cwe,[cwe], issue.lineNumber)
			issueList.append(existingScannerIssue)
		
		self.foundIssues['JULIET'] = IssueComparisonResult('JULIET', issueList)
			
	
	# add an issues found by one of the scanners
	def addFoundScannerIssues(self, securityScanner, foundIssues):
		self.foundIssues[securityScanner] = IssueComparisonResult(securityScanner, foundIssues)
	
	# compare the found ones with the existing ones
	def compareIssues(self):
		if(len(self.existingIssues)>0):
			for scanner, issueHolder in self.foundIssues.items():

				for testIssue in issueHolder.foundIssues:
				
					issueHolder.issueCnt+=1
					correctMatchCnt=False
					differentTypeMatches=False
					rangeMatch=False
					differentLineMatches=False
					noneMatching=False
					for issue in self.existingIssues:
						#AW20130722 if more than one cwe's are possible check which one is found
						if(len(testIssue.cweList)>1):
							for cweEntry in testIssue.cweList:
								if(issue.category=="CWE"+cweEntry):
									testIssue.cwe = cweEntry
									
					
						if issue.lineNumber == testIssue.lineNumber and issue.category == "CWE"+testIssue.cwe and issue.endLine!=-1:
							correctMatchCnt=True
						elif issue.lineNumber==testIssue.lineNumber and issue.category!="CWE"+testIssue.cwe and issue.endLine!=-1:
							differentTypeMatches = True
						# elif issue.startLine!=-1 and issue.endLine!=-1 and issue.category=="CWE"+testIssue.cwe and testIssue.lineNumber >= issue.startLine and testIssue.lineNumber <= issue.endLine:
						elif issue.startLine!=-1 and issue.endLine!=-1 and issue.category=="CWE"+testIssue.cwe:
							rangeMatch=True
						elif issue.category=="CWE"+testIssue.cwe:
							differentLineMatches=True
						else:
							noneMatching=True
					
					if(correctMatchCnt):
						issueHolder.correctMatchCnt+=1
					elif(differentTypeMatches):
						issueHolder.differentTypeMatches+=1
					elif(rangeMatch):
						issueHolder.rangeMatch+=1
					elif(differentLineMatches):
						issueHolder.differentLineMatches+=1
					else:
						issueHolder.noneMatching+=1
		else:
			for scanner, issueHolder in self.foundIssues.items():
				issueHolder.noneMatching+=len(issueHolder.foundIssues)	

