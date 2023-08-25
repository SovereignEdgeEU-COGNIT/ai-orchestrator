class EdgeCluster:
    def __init__(self, name):
        self._name = name
        self._hosts = {}

    @property
    def name(self):
        return self._name

    def add_host(self, host):
        self._hosts[host.name] = host 

