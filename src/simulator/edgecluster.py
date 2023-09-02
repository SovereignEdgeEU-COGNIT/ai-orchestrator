from prometheus_client import Gauge
from energymix import *

class EdgeCluster:
    def __init__(self, name):
        self._name = name
        self._servers = {}
        self._energymix = EnergyMix()
        self._cpuload_gauge = Gauge("edgecluster_cpuload_"+name, "cpuload", ["server"])
        self._processes_gauge = Gauge("edgecluster_processes_"+name, "processes", ["server"])
        self._energymix_gauge = Gauge("edgecluster_energymix_"+name, "energymix", ["energytype"])
        self._energymix_state_gauge = Gauge("edgecluster_energymix_state_"+name, "energymix_state")

    @property
    def name(self):
        return self._name
    
    @property
    def servers(self):
        return list(self._servers.values())
    
    @property
    def cpuload_gauge(self):
        return self._cpuload_gauge
    
    @property
    def processes_gauge(self):
        return self._processes_gauge
    
    @property
    def energymix_gauge(self):
        return self._energymix_gauge
    
    @property
    def energymix_state_gauge(self):
        return self._energymix_state_gauge
    
    @property
    def energymix(self):
        return self._energymix
    
    def add_server(self, server):
        self._servers[server.name] = server 

