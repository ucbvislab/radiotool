
# constraints should be multiplicative so we can do them in any order
# ok, maybe not multiplicative. not sure yet.
# want to avoid plateaus in the space.

import copy

import numpy as np

import librosa_analysis
import novelty

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
        self.in_labels = copy.copy(in_labels)
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
                    new_pen[n_i, l] = 0.0
                elif node_label is None:
                    # should this have a penalty?
                    new_pen[n_i, l] = 0.0

                if self.window > 0:
                    if target_label != prev_target:
                        # reduce penalty for beats prior
                        span = min(self.window, l)
                        new_pen[n_i, l - span:l] = np.linspace(1.0, 0.01, num=span)

                    if target_label != next_target:
                        # reduce penalty for beats later
                        span = min(self.window, len(self.out_labels) - l - 1)
                        new_pen[n_i, l + 1:l + span + 1] = np.linspace(0.01, 1.0, num=span)

            for l in [0, n_target - 1]:
                target_label = self.out_labels[l]
                if node_label == target_label or target_label is None:
                    new_pen[n_i, l] = 0.0
                elif node_label is None:
                    new_pen[n_i, l] = 0.0
        
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
        self.to_cost = .075
        self.bw_cost = .05

    def apply(self, transition_cost, penalty, song):
        # we have to manage the pauses...
        n_beats = len(song.analysis["beats"])
        beat_len = song.analysis["avg_beat_duration"]
        min_beats = int(np.ceil(self.min_len / float(beat_len)))
        max_beats = int(np.floor(self.max_len / float(beat_len)))

        tc = self.to_cost * min_beats
        bc = self.bw_cost

        new_trans = np.zeros((n_beats + max_beats, n_beats + max_beats))
        new_trans[:n_beats, :n_beats] = transition_cost
        
        new_pen = np.zeros((n_beats + max_beats, penalty.shape[1]))
        new_pen[:n_beats, :] = penalty

        # beat to first pause
        p0 = n_beats
        p_n = p0 + max_beats - 1
        new_trans[:n_beats, p0] = tc
        
        # beat to other pauses
        new_trans[:n_beats, p0 + 1:] = np.inf

        # pause to pause default
        new_trans[p0:, p0:] = np.inf
        
        # must stay in pauses until min pause
        for i in range(p0, p0 + min_beats):
            new_trans[i, :n_beats] = np.inf
            new_trans[i, i + 1] = 0.
        
        # after that, pause-to-pause costs something
        for i in range(p0 + min_beats, p0 + max_beats - 2):
            new_trans[i, :n_beats] = np.inf
            # new_trans[i, :n_beats] = 0.
            new_trans[i, p_n] = 0.
            new_trans[i, i + 1] = bc

        # last pause must go back to beats
        # Also, must exit through last pause
        new_trans[p_n, :n_beats] = 0.

        new_pen[p0 + 1:, 0] = np.inf

        return new_trans, new_pen


class PauseEntryConstraint(Constraint):
    def __init__(self, target_labels, penalty_value):
        self.out_labels = target_labels
        self.p = penalty_value

    def apply(self, transition_cost, penalty, song):
        n_beats = len(song.analysis["beats"])
        n_pauses = transition_cost.shape[0] - n_beats
        p0 = n_beats
        
        if n_pauses > 0:
            print "Pause entry constraints"
            target_changes = [0]
            for l in xrange(1, len(self.out_labels)):
                target = self.out_labels[l]
                prev_target = self.out_labels[l - 1]
                if target != prev_target:
                    target_changes.append(l)

            target_changes = np.array(target_changes)

            penalty[p0, :] += self.p
            penalty[p0, target_changes] -= self.p

        return transition_cost, penalty


class PauseExitConstraint(Constraint):
    def __init__(self, target_labels, penalty_value):
        self.out_labels = target_labels
        self.p = penalty_value

    def apply(self, transition_cost, penalty, song):
        n_beats = len(song.analysis["beats"])
        if transition_cost.shape[0] > n_beats:
            print "Pause exit constraints"
            p_n = transition_cost.shape[0] - 1
            target_changes = [0]
            for l in xrange(1, len(self.out_labels)):
                target = self.out_labels[l]
                prev_target = self.out_labels[l - 1]
                if target != prev_target:
                    target_changes.append(l)


            target_changes = np.array(target_changes)

            penalty[p_n, :] += self.p
            penalty[p_n, target_changes] -= self.p

        return transition_cost, penalty


class NoveltyConstraint(Constraint):
    def __init__(self, in_labels, target_labels, penalty):
        self.in_labels = in_labels
        self.out_labels = target_labels
        self.penalty = penalty

    def apply(self, transition_cost, penalty, song):
        changepoints = np.array(novelty.novelty(song))
        beats = song.analysis["beats"]
        n_beats = len(beats)
        n_target = penalty.shape[1]
        cp_beats_i = [np.argmin(np.abs(beats - cp)) for cp in changepoints]
        cp_beats = [beats[i] for i in cp_beats_i]

        # find emotional changes at each changepoint, if any
        changes = []
        for i in cp_beats_i:
            # check the previous and next 4 beats
            n_prev = min(4, i)
            n_next = min(4, n_beats - i)
            labs = [self.in_labels[j]
                    for j in range(i - n_prev, i + n_next + 1)]
            # check first and last beat in this range... assuming a sort of
            # coarse-grained emotional labeling
            if labs[0] != labs[-1]:
                # there is an emotional change at this point in the music
                changes.append((i, labs[0], labs[-1]))

        for change in changes:
            print "Found emotional change near changepoint: " + change[1] + " -> " + change[2]

        # find those emotional changes in the target output
        for l in xrange(1, n_target):
            target = self.out_labels[l]
            prev_target = self.out_labels[l - 1]
            if target != prev_target:
                for change in changes:
                    if prev_target == change[1] and target == change[2]:
                        print "setting change:\t" + change[1] + " -> " + change[2] 
                        print "\tat beat " + str(l) + " " + str(l * song.analysis["avg_beat_duration"])

                        # give huge preference to hitting the changepoint here
                        beat_i = change[0]
                        penalty[:n_beats, l] += 1.0
                        n_prev = min(2, beat_i)
                        n_next = min(2, n_beats - beat_i)
                        penalty[beat_i - n_prev:beat_i + n_next, l] -= 1.0

        return transition_cost, penalty


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

