[Korean](./README(Korean).md)

# Tools Configuration
- Tools Configuration involves selecting and using multiple static analysis to identify potential vulnerabilities in C/C++ source code.
- The architecture of this component is shown below:

<p align="center">
  <img src="./docs/tools.png" />
  <span>Architecture for Line-Level Vulnerability Analysis Using BERT</span>
</p>

## File Structure
- The following file structure is used to implement this model
```
- data: Folder for storing dataset
- tmpData: Folder for storing tool output results
- tools-usage: Usage documentation for each analysis tool
- docker: Dockerfiles to run the tools in a containerized environment
- AnalyzeToolConfig: Parses configuration files and sets tool parameters and options
  - buildScannerList()
  - getCCppScannerList()
- CompareTool: Compares tool outputs against known vulnerabilities from the Juliet test suite
- ComparisonResultHolder: Simple result holder class used for summary comparison
- config.cfg: Configuration file for the entire project
- convertTool: Converts outputs from different static analyzers into a unified internal format
- FlawCollector: Collects vulnerabilities based on rules from Juliet test suite
- HTMLReport: Generates HTML-formatted output reports
- Issue: Represents issues in XML format using Issue tag classes
- IssueComparisionResult: Class for managing issue information
- IssueComparison: Class for comparing issue
- metricGenerator: Generates evaluation metrics
- MSCompilerResultConverter: Converts outputs
- py_common: Utility functions
- [main] runCompleteAnalysis: Main function to start analysis 
- ScannerCWEMapping: Maps scanner results to CWE IDs
- ScannerIssueHolder: Lightweight issue holder used during comparisons
- SecurityModel: Models detected vulnerabilities
- SecurityModelComparision: Compared detected vulnerabilities
- SecurityScanner: Encapsulates various security scanners based on configuration
- TransformTool
- TestsuiteAnalyzer: Main script to configure and run analysis tools based on test filenames from the Juliet suite
```

## Setup & Execution
Each tool is executed as follow:
1. Infer
- Use `Makefile` to build and analyze each test case
2. Clang
- Run `clang.sh` to analyze each test case with customized options
3. Cppcheck
- Execute via `cppcheck` command line
4. Flawfinder
- Execute via `cppcheck` command line
5. Framac
- Execute via `frama-c` command line
6. PVS- Studio
- Use `pvs_studio.sh` to analyze each test case

### Final Execution Steps
1. Prepare Executables for Analysis Tools
- Download the Juliet C/C++ dataset from the julietsuite folder
- Run `python3 handle_make.py`
  - Infer
  - Clang
  - Cppcheck 

2. Fun Full Toolchain Analysis
```
python3 TestsuiteAnalyzer.py
```
- This command will automatically analyze each Juliet test case using all supported tools and process the output accordingly.