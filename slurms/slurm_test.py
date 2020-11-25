import multiprocessing
import time
import subprocess
import psutil
import os

#  text = subprocess.run(["top", "-p", "1", "-n", "1"], stdout=subprocess.PIPE)
#  print(text.stdout.decode("utf-8"))
print(multiprocessing.cpu_count())
print(psutil.cpu_count())
print(os.cpu_count())
print(len(os.sched_getaffinity(0)))
