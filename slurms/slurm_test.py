import multiprocessing
import time

for i in range(0, 100):
    print(multiprocessing.cpu_count())
    time.sleep(0.5)
