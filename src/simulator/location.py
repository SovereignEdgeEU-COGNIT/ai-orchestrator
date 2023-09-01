class Location:
    def __init__(self, name):
        self._name = name
        self._edgeclusters = {}

    @property
    def name(self):
        return self._name
    
    @property
    def edgeclusters(self):
        return list(self._edgeclusters.values())

    def add_edgecluster(self, edgecluster):
        self._edgeclusters[edgecluster.name] = edgecluster

