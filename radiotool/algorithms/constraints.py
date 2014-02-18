
# constraints should be multiplicative so we can do them in any order
# ok, maybe not multiplicative. not sure yet.
# want to avoid plateaus in the space.

import numpy as np

import librosa_analysis


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
    def __init__(self): pass

    def apply(self, transition_cost, penalty, song): return transition_cost, penalty


class TimbrePitchConstraint(Constraint):
    def __init__(self, timbre_weight=1, chroma_weight=1):
        self.timbre_weight = timbre_weight
        self.chroma_weight = chroma_weight

    def apply(self, transition_cost, penalty, song):
        timbre_dist = librosa_analysis.structure(np.array(song.analysis['timbres']).T)
        chroma_dist = librosa_analysis.structure(np.array(song.analysis['chroma']).T)

        tw = float(self.timbre_weight) / (float(self.chroma_weight) + float(self.timbre_weight))
        cw = 1 - tw

        dists = tw * timbre_dist + cw * chroma_dist

        # dists = np.copy(song.analysis["dense_dist"])
        # shift it over
        dists[:-1, :] = dists[1:, :]
        dists[-1, :] = np.inf
        transition_cost[:dists.shape[0], :dists.shape[1]] *= dists
        return transition_cost, penalty


class RhythmConstraint(Constraint):
    def __init__(self, beats_per_measure, multiplier):
        self.m = multiplier
        self.time = beats_per_measure

    def apply(self, transition_cost, penalty, song):
        n_beats = len(song.analysis["beats"])
        for i in range(self.time):
            for j in set(range(self.time)) - set([(i + 1) % self.time]):
                transition_cost[i:n_beats:self.time][j:n_beats:self.time] *= self.m
        return transition_cost, penalty


class MinimumJumpConstraint(Constraint):
    def __init__(self, min_jump):
        self.min_jump = min_jump

    def apply(self, transition_cost, penalty, song):
        n_beats = len(song.analysis["beats"])
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
        n_beats = len(song.analysis["beats"])

        # extend in_labels to work with pauses that we may have added
        if n_beats < transition_cost.shape[0]:
            self.in_labels.extend([None] * (transition_cost.shape[0] - n_beats))

        new_pen = np.ones(penalty.shape) * np.array(self.penalty)
        # new_pen = np.ones((n_beats, len(self.penalty))) * np.array(self.penalty)
        n_target = penalty.shape[1]
        for n_i in xrange(transition_cost.shape[0]):
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
        
        penalty += new_pen

        return transition_cost, penalty


class GenericTimeSensitivePenalty(Constraint):
    def __init__(self, penalty):
        self.penalty = penalty

    def apply(self, transition_cost, penalty, song):
        penalty[:n_beats, :] += self.penalty
        return transition_cost, penalty


class PauseConstraint(Constraint):
    def __init__(self, min_length, max_length):
        self.min_len = min_length
        self.max_len = max_length
        self.to_cost = .25
        self.bw_cost = .05

    def apply(self, transition_cost, penalty, song):
        # we have to manage the pauses...
        n_beats = len(song.analysis["beats"])
        beat_len = song.analysis["avg_beat_duration"]
        min_beats = int(np.ceil(self.min_len / float(beat_len)))
        max_beats = int(np.floor(self.max_len / float(beat_len)))

        new_trans = np.zeros((n_beats + max_beats, n_beats + max_beats))
        new_trans[:n_beats, :n_beats] = transition_cost
        
        new_pen = np.zeros((n_beats + max_beats, penalty.shape[1]))
        new_pen[:n_beats, :] = penalty

        # beat to first pause
        p0 = n_beats
        new_trans[:n_beats, p0] = self.to_cost
        
        # beat to other pauses
        new_trans[:n_beats, p0 + 1:] = np.inf

        # pause to pause default
        new_trans[p0:, p0:] = np.inf
        
        # must stay in pauses until min pause
        for i in range(p0, p0 + min_beats):
            new_trans[i, :n_beats] = np.inf
            new_trans[i, i + 1] = 0.
        
        # after that, pause-to-pause costs something
        for i in range(p0 + min_beats, p0 + max_beats - 1):
            new_trans[i, :n_beats] = 0.
            new_trans[i, i + 1] = self.bw_cost

        # last pause must go back to beats
        new_trans[p0 + max_beats - 1, :n_beats] = 0.

        new_pen[p0 + 1:, 0] = np.inf

        return new_trans, new_pen


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

