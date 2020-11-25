import random
import multiprocessing
from joblib import Parallel, delayed
import time
import subprocess

#  import psutil
import os

#  text = subprocess.run(["top", "-p", "1", "-n", "1"], stdout=subprocess.PIPE)
#  print(text.stdout.decode("utf-8"))
print(multiprocessing.cpu_count())
#  print(psutil.cpu_count())
print(os.cpu_count())
print(len(os.sched_getaffinity(0)))


def run_parallel(i):
    print(i)
    time.wait(random.randint(0, 1))


with Parallel(
    multiprocessing.cpu_count(),
    backend="threading",
) as parallel:
    output = parallel(delayed(run_parallel)(i) for i in range(1, 10))
