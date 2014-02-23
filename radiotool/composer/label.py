
class Label(object):
    def __init__(self, name, time):
        self.name = name
        self.time = time

    def sample(self, samplerate):
        return int(samplerate * self.time)

    def __repr__(self):
        return "Label %s at time %f" % (self.name, self.time)
