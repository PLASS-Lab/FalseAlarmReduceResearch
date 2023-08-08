import os
import sys
from os import listdir
from os.path import isfile, join
from subprocess import Popen, PIPE
import shlex
import time

# simply run a system command and return the needed info
def system_call(cmd, dir="."):
    print("[CMD]:", cmd)
    vtime = time.time()
    process = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, cwd='.')
    (output, err) = process.communicate()
    exit_code = process.wait()
    if (exit_code == 0):
        print("[DONE]", "\n")
    else:
        print("[FAILED]:", err.decode("utf-8"))
        print("[EXIT CODE]:", exit_code)
    print("[OUTPUT]:", output.decode("utf-8"))
    vtime = round(time.time() - vtime, 3);
    return (output, err, exit_code, vtime)

def list_files(directory, extension, absolute_path=False):
    files = []

    for f in listdir(directory):
        fpath = join(directory, f)
        if isfile(fpath) and fpath.endswith(extension):
            files.append(fpath)
    return files

def append_in(file_path, text):
    sys.stdout = open(file_path, "a")
    print(text)
    sys.stdout = sys.__stdout__

def reset_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    sys.stdout = open(file_path, "w")
    sys.stdout = sys.__stdout__

def file_line_error_header(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    sys.stdout = open(file_path,"w")
    print("File, Line, Error, CWE")
    sys.stdout = sys.__stdout__
    
def tool_exec_log(file_path, cmd, out, err, exit):
    sys.stdout = open(file_path, "a")
    print("[CMD]: " + cmd) 
    print("[OUTPUT]:\n" + out.decode("utf-8"))
    print("[ERR]:\n" + err.decode("utf-8"))
    print("[EXIT]: " + str(exit) + "\n")
    sys.stdout = sys.__stdout__
    
if __name__ == '__main__':
    csv = './temp.csv'
    file_line_error_header(csv)
    # cmd = 'clang++ -cc1 -analyze -analyzer-checker=core,alpha  -I ./1v3/juliet_suite-c-cplus/src/testcasesupport -I /usr/include -I /usr/include/x86_64-linux-gnu/ -I /usr/lib/clang/6.0/include ./1v3/juliet_suite-c-cplus/testcases/CWE366_Race_Condition_Within_Thread/CWE366_Race_Condition_Within_Thread__global_int_01.c'
    # cmd = "frama-c -val -kernel-log ew:tmpData/framac_temp.log /home/huong/projects/VDCT/1v3/juliet_suite-c-cplus/testcases/CWE617_Reachable_Assertion/main_linux.cpp -cpp-extra-args='-I./1v3/juliet_suite-c-cplus/src/testcasesupport  -DINCLUDEMAIN -U__cplusplus'"
    cmd = "frama-c -quiet -main CWE617_Reachable_Assertion__connect_socket_01_bad /home/huong/projects/VDCT/1v3/juliet_suite-c-cplus/testcases/CWE617_Reachable_Assertion/CWE617_Reachable_Assertion__connect_socket_01.c -cpp-extra-args='-I./1v3/juliet_suite-c-cplus/src/testcasesupport  -DINCLUDEMAIN -U__cplusplus'"
    (output, err, exit_code, vtime) = system_call(cmd, ".")
    tool_exec_log('./temp.txt', cmd, output, err, exit)
    sys.stdout = open(csv, "a")
    print(err, file=sys.__stdout__)
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

    print(err)