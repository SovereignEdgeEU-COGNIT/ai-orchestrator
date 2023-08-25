class Location:
    def __init__(self, name):
        self._name = name
        self._edgeclusters = {}

    @property
    def name(self):
        return self._name

    def add_edgecluster(self, edge_cluster):
        self._edgeclusters[edge_cluster.name] = edge_cluster

