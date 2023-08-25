import unittest
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

        for i in range(25):
            computer.execute()
            print(computer.cpuload)
            clock.tick_ms()

