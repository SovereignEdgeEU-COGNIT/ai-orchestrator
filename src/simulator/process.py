from scipy.stats import beta
import numpy as np

# Categories:
# cpu_bound:
#   e.g. heavy mathematical computations, graphics rendering, simulations, etc.
# mem_bound: 
#   e.g. databases storing/retrieving vast amounts of data, large-scale data processing tasks, etc.
# io_bound:
#   e.g. web servers, file servers, database servers (if the primary bottleneck is reading/writing to disk), etc.
# net_bound:
#   e.g. online multiplayer games, streaming servers, etc.
# disk_bound:
#   e.g large-scale data transformation jobs, applications which require frequent reading/writing of large files, etc.
# gpu_bound:
#   e.g. rendering software, machine learning training tasks, video processing tools, etc.

class Process:
    def __init__(self, exectime=0, category=[]):
        self._pid = -1
        self._exectime = exectime
        self._category = category
        self._cpuload = 0.0
        self._memload = 0.0
        self._ioload = 0.0
        self._netload = 0.0
        self._gpuload = 0.0
        self._starttime = 0
        self._started = False

    @property
    def exectime(self):
        return self._exectime
    
    @property
    def started(self):
        return self._started
    
    @property
    def pid(self):
        return self._pid

    def start(self, timestamp):
        self._started = True
        self._starttime = timestamp
   
    def set_pid(self, pid):
        self._pid = pid

    @property
    def cpuload(self):
        return self._cpuload
    
    def duration(self, now):
        return now - self._starttime
    
    def execute(self):
        if self._started is False:
            return 
        for c in self._category:
            if c == "cpu_bound":
                a, b = 20, 2 
                self._cpuload = beta.rvs(a, b, size=1)
            else:
                a, b = 2, 20 
                self._cpuload = beta.rvs(a, b, size=1)
        if len(self._category) == 0:
            a, b = 2, 20 
            self._cpuload = beta.rvs(a, b, size=1)
