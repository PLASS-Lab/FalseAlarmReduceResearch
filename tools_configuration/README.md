Vulenerability Dection Combination Tools Project

curl --request POST --silent -u adminadmin http127.0.0.19000apiuser_tokenssearch

GCC
httpsseamless.tistory.com2
https://gcc.gnu.org/onlinedocs/gcc/Overall-Options.html
https://gcc.gnu.org/onlinedocs/gcc/Option-Summary.html

Makefile
httpseslinuxprogramming.blogspot.com201504gnu-make.html

## Project Structure
The folder structure of this app is explained below
- data  데이터를 저장하는 폴더
- tmp Data  도구를 통해서 결과를 저장하는 폴더
- tools-usage 도구별로 사용법 파일
- docker  Docker 환경을 실행하기 위해서 Docker 파일
- AnalyzeToolConfig config 파일로 부터 설정한 정보를 받아서 도구를 paramater 및 option 설정
  - buildScannerList()
  - getCCppScannerList()
- CompareTool 도구는 testsuite의 기존 문제 기반으로 진단되는 출력 결과을 비교함
- ComparisonResultHolder Summary 비교하는 데 사용되는 단순한 일반 결과 홀더
- config.cfg 프로젝트 Config 정보를 저장하는 파일
- convertTool 여러 보안 검사기의 출력을 자체 형식으로 변환하는 도구 추가 처리에 필요함, 즉 출력되는 결과의 양식을 변환
- CppcheckResultConverter CppCheck라는 도구는 결과를 출력되었으면 해당 결과를 변화
- FlawCollector 처음에 juliet-suite 데이터를 받아서 취약점을 규칙 수집함
- HTMLReport 결과는 html 형식으로 변화해주는 파일 
- Issue Xml 파일 저장하기 위해서 Issue tag 클래스
- IssueComparisionResult Issue 정보를 만든 클래스
- IssueComparison Issue를 비교하는 클래스
- metricGenerator metric를 생성하는 클래스
- MSCompilerResultConverter
- py_common 파일을 작업 등 함수
- ResultConverter 
  - getXML()
- [main] runCompleteAnalysis main 함수 (실행 시작) 
- ScannerCWEMapping 스캐너 결과 cwe 매핑을 캡슐화하는 단순 클래스
- ScannerIssueHolder 전체 이슈 클래스보다 가볍움. 이슈 비교에 필요
- SecurityModel 시코딩 모델
- SecurityModelComparision  시코딩 모델 비교
- SecurityScanner 다양한 종류의 보안 스캐너를 캡슐화하기 위한 간단한 클래스, 즉 config에 기준으로 정보를 추가함
- TransformTool
- TestsuiteAnalyzer juliet-test-filename'에서 설정을 되는 분석 도구 run_analysis_tool