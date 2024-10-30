import os
import subprocess
ida_path = "D:/Program Files/IDA_Pro_v7.5/ida.exe"
work_dir = os.path.abspath('.')
#elfpath = os.path.join(work_dir, 'binaryfile/libcurl.so.4.6.0')
elfpath = os.path.join(work_dir, 'binaryfile/binutils-strings-2.34-O3')
#script_path = os.path.join(work_dir, "Genius/raw-feature-extractor_py3_ida7.7/preprocessing_ida.py")
#script_path = os.path.join(work_dir, "./analysis.py")
script_path = os.path.join(work_dir, "./getFuncAsm.py")
# cmd_str = ida.exe -Lida.log -c -A -Sanalysis.py pefile
cmd_str = '{} -Lida.log -c -A -S{} {}'.format(ida_path, script_path, elfpath)
print(cmd_str)

p = subprocess.Popen((cmd_str))
p.wait()