#!/bin/bash

echo $1
path_name=$1
path_directory=/home/huong/projects/VDCT/tmpData/CCPP/clang/$path_name
echo $path_directory

mkdir -p $path_directory

for NAME in `find . -name "CWE*.c"`
do
	echo ${NAME%.c}
	clang -Wno-unused-command-line-argument  -I/home/huong/projects/VDCT/1v3/juliet_suite-c-cplus/src/testcasesupport --analyze -Xanalyzer -analyzer-checker=alpha,core $NAME  -o $path_directory/${NAME%.c}.xml
done

for NAME in `find . -name "CWE*.cpp"`
do
	echo ${NAME%.cpp}
	clang++ -Wno-unused-command-line-argument  -I/home/huong/projects/VDCT/1v3/juliet_suite-c-cplus/src/testcasesupport --analyze -Xanalyzer -analyzer-checker=alpha,core $NAME  -o $path_directory/${NAME%.cpp}.xml
done
