import unittest
import time
from location import *
from edgecluster import *
from computer import *
from clock import *
from process import *

class TestSimulator(unittest.TestCase):
    def test_location(self):
        clock = Clock()
        
        loc = Location("test_name")
        edgecluster = EdgeCluster("test_name")
        loc.add_edgecluster(edgecluster)
        
        computer = Computer("test_name", clock, cores=1)
        edgecluster.add_host(computer)
        
        process = Process(exectime=20*1e6, category=[])  # run for 20 milliseconds
        #process = Process(exectime=20*1e6, category=["cpu_bound", "io_bound"])  # run for 20 milliseconds
        computer.launch(process)

        for i in range(1000):
            mu, sigma = 0, 0.1 # mean and standard deviation
            a, b = 10, 10 
            s = beta.rvs(a, b, size=1)[0]
            if s > 0.8:
                process = Process(exectime=s*1e6*200, category=[])
                computer.launch(process)
                print("starting process <{}>".format(process.pid))
            computer.execute()
            print(computer.cpuload)
            clock.tick_ms()
            time.sleep(1/1000)
