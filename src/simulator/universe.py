class Universe:
    def __init__(self):
        self._locations = []

    @property
    def locations(self):
        return self._locations
                    
    def add_location(self, loc):
        self._locations.append(loc)
