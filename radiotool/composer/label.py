
class Label(object):
    def __init__(self, name, time):
        self.name = name
        self.time = time

    def sample(self, samplerate):
        return int(samplerate * self.time)
