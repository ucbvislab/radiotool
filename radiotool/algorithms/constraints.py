
# constraints should be multiplicative so we can do them in any order

import numpy as np


class ConstraintPipeline(object):
    def __init__(self, constraints=None):
        if constraints is None:
            self.constraints = []
        else:
            self.constraints = constraints

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def apply(self, song, target_n_length):
        n_beats = len(song.analysis["beats"])
        transition_cost = np.ones((n_beats, n_beats))
        penalty = np.zeros((n_beats, target_n_length))
        for constraint in self.constraints:
            transition_cost, penalty = constraint.apply(transition_cost, penalty, song)
        return transition_cost, penalty


class Constraint(object):
    def __init__(self):
        pass

    def apply(self, transition_cost, penalty, song):
        return transition_cost


class TimbrePitchConstraint(Constraint):
    def apply(self, transition_cost, penalty, song):
        dists = np.copy(song.analysis["dense_dist"])
        # shift it over
        dists[:-1, :] = dists[1:, :]
        dists[-1, :] = np.inf
        return transition_cost * dists, penalty


class RhythmConstraint(Constraint):
    def __init__(self, beats_per_measure):
        self.time = beats_per_measure

    def apply(self, transition_cost, penalty, song):
        for i in range(self.time):
            for j in set(range(self.time)) - set([(i + 1) % self.time]):
                transition_cost[i::self.time][j::self.time] *= 2.0
        return transition_cost, penalty


class MinimumJumpConstraint(Constraint):
    def __init__(self, min_jump):
        self.min_jump = min_jump

    def apply(self, transition_cost, penalty, song):
        n_beats = transition_cost.shape[0]
        for i in range(n_beats):
            for j in range(-(self.min_jump - 1), self.min_jump):
                if 0 < i + j < n_beats and j != 1:
                    transition_cost[i, i + j] = np.inf
        return transition_cost, penalty


class LabelConstraint(Constraint):
    def __init__(self, in_labels, target_labels, penalty, penalty_window=0):
        self.in_labels = in_labels
        self.out_labels = target_labels
        self.penalty = penalty
        self.window = penalty_window

    def apply(self, transition_cost, penalty, song):

        new_pen = np.ones(penalty.shape) * np.array(self.penalty)

        n_beats = transition_cost.shape[0]
        n_target = penalty.shape[1]
        for n_i in xrange(n_beats):
            node_label = self.in_labels[n_i]
            for l in xrange(1, n_target - 1):
                prev_target = self.out_labels[l - 1]
                next_target = self.out_labels[l + 1]
                target_label = self.out_labels[l]

                if node_label == target_label or target_label is None:
                    new_pen[n_i, l] = 0

                if self.window > 0:
                    if target_label != prev_target:
                        # reduce penalty for beats prior
                        span = min(self.window, l)
                        new_pen[n_i, l - span:l] = N.linspace(1.0, 0.01, num=span)

                    if target_label != next_target:
                        # reduce penalty for beats later
                        span = min(self.window, len(self.out_labels) - l - 1)
                        new_pen[n_i, l + 1:l + span + 1] = N.linspace(0.01, 1.0, num=span)

            for l in [0, n_target - 1]:
                target_label = self.out_labels[l]
                if node_label == target_label or target_label is None:
                    new_pen[n_i, l] = 0

        return transition_cost, penalty + new_pen


class GenericTimeSensitivePenalty(Constraint):
    def __init__(self, penalty):
        self.penalty = penalty

    def apply(self, transition_cost, penalty, song):
        return transition_cost, penalty + self.penalty


if __name__ == '__main__':
    import sys
    from radiotool.composer import Song
    song = Song(sys.argv[1])

    pipeline = ConstraintPipeline(constraints=[
        TimbrePitchConstraint(),
        RhythmConstraint(4),
        MinimumJumpConstraint(4)
    ])

    tc, penalty = pipeline.apply(song, 102)

