import unittest
from clock import *

class TestClock(unittest.TestCase):
    def test_lock(self):
        clock = Clock()
        #print(clock.display())
        clock.tick_ns(10 * 1e9)  # Tick by 10 seconds worth of nanoseconds
        #print(clock.display())
        clock.set_time(0)  # Set to start of Unix time
        #print(clock.display())
