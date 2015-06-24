__author__ = 'cclamb'


class Prop(object):

    def __init__(self):
        self._value = 0

    @property
    def x(self):
        return self._value

    @x.setter
    def x(self, x):
        self._value = x