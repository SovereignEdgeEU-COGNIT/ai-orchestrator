from prometheus_client import Gauge

class EdgeCluster:
    def __init__(self, name):
        self._name = name
        self._servers = {}
        self._cpuload_guage = Gauge("edgecluster_"+name, "cpuload", ["server"])

    @property
    def name(self):
        return self._name
    
    @property
    def servers(self):
        return list(self._servers.values())
    
    @property
    def cpuload_gauge(self):
        return self._cpuload_guage

    def add_server(self, server):
        self._servers[server.name] = server 

