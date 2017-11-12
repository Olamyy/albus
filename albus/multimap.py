class MultiMap(object):
    def __init__(self, key, **values):
        self.key = key
        self.values = values
        self.map = {self.key: self.values}

    def __str__(self):
        return str(self.map)

    def __repr__(self):
        return "A multimap : {0}".format(self.map)

    def __eq__(self, multimap):
        if self.map.keys() == multimap.keys():
            return self.map.values == multimap.values()
        return False

    def __le__(self, multimap):
        return len(self.map.keys()) < len(multimap.keys())

    def __gt__(self, multimap):
        return len(self.map.keys()) > len(multimap.keys())
