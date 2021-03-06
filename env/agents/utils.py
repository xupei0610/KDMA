class Vec2():
    def __init__(self, x, y):
        self._x = x
        self._y = y
    def __getitem__(self, idx):
        if idx == 0:
            return self.x
        if idx == 1:
            return self.y
        raise IndexError(idx)

    @property
    def x(self):
        return self._x
    @property
    def y(self):
        return self._y
    
    def __str__(self):
        return "x: {}, y: {}".format(self.x, self.y)
