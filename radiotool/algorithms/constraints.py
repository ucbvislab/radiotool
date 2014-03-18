
# constraints should be multiplicative so we can do them in any order
# ok, maybe not multiplicative. not sure yet.
# want to avoid plateaus in the space.

import copy

import numpy as np
from scipy.special import binom
import scipy.spatial.distance

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
        beat_names = copy.copy(song.analysis["beats"])
        transition_cost = np.ones((n_beats, n_beats))
        penalty = np.zeros((n_beats, target_n_length))
        for constraint in self.constraints:
            print constraint
            transition_cost, penalty, beat_names = constraint.apply(
                transition_cost, penalty, song, beat_names)
        return transition_cost, penalty, beat_names


class Constraint(object):
    def __init__(self): pass

    def apply(self, transition_cost, penalty, song, beat_names):
        return transition_cost, penalty, beat_names


class RandomJitterConstraint(Constraint):
    def __init__(self, jitter_max=.001):
        self.jitter = jitter_max

    def apply(self, transition_cost, penalty, song, beat_names):
        return (
            transition_cost + self.jitter * np.random.rand(
                transition_cost.shape[0], transition_cost.shape[1]),
            penalty + self.jitter * np.random.rand(
                penalty.shape[0], penalty.shape[1]),
            beat_names)


class TimbrePitchConstraint(Constraint):
    def __init__(self, timbre_weight=1, chroma_weight=1, context=1):
        self.tw = float(timbre_weight) / (float(chroma_weight) + float(timbre_weight))
        self.cw = 1 - self.tw
        self.m = context

    def apply(self, transition_cost, penalty, song, beat_names):
        timbre_dist = librosa_analysis.structure(np.array(song.analysis['timbres']).T)
        chroma_dist = librosa_analysis.structure(np.array(song.analysis['chroma']).T)

        dists = self.tw * timbre_dist + self.cw * chroma_dist

        if self.m > 1:
            new_dists = np.zeros(dists.shape)
            coefs = [binom(self.m * 2, i) for i in range(self.m * 2 + 1)]
            coefs = np.array(coefs) / np.sum(coefs)
            for beat_i in xrange(dists.shape[0]):
                for beat_j in xrange(dists.shape[1]):
                    entry = 0.0
                    for i, c in enumerate(coefs):
                        t = i - self.m
                        if beat_i + t >= 0 and beat_i + t < dists.shape[0] and\
                            beat_j + t >= 0 and beat_j + t < dists.shape[1]:
                            entry += c * dists[beat_i + t, beat_j + t]
                    new_dists[beat_i, beat_j] = entry

            dists = new_dists

        # dists = np.copy(song.analysis["dense_dist"])
        # shift it over
        dists[:-1, :] = dists[1:, :]
        dists[-1, :] = np.inf

        # don't use the final beat
        dists[:, -1] = np.inf
        
        transition_cost[:dists.shape[0], :dists.shape[1]] *= dists

        return transition_cost, penalty, beat_names

    def __repr__(self):
        return "TimbrePitchConstraint: %f(timbre) + %f(chroma)" % (self.tw, self.cw)


class RhythmConstraint(Constraint):
    def __init__(self, beats_per_measure, multiplier):
        self.m = multiplier
        self.time = beats_per_measure

    def apply(self, transition_cost, penalty, song, beat_names):
        n_beats = len(song.analysis["beats"])
        for i in range(self.time):
            for j in set(range(self.time)) - set([(i + 1) % self.time]):
                transition_cost[i:n_beats:self.time][j:n_beats:self.time] *= self.m
        return transition_cost, penalty, beat_names


class MinimumJumpConstraint(Constraint):
    def __init__(self, min_jump):
        self.min_jump = min_jump

    def apply(self, transition_cost, penalty, song, beat_names):
        n_beats = len(song.analysis["beats"])
        for i in range(n_beats):
            for j in range(-(self.min_jump - 1), self.min_jump):
                if 0 < i + j < n_beats and j != 1:
                    transition_cost[i, i + j] = np.inf
        return transition_cost, penalty, beat_names

    def __repr__(self):
        return "MinimumJumpConstraint: min_jump(%d)" % self.min_jump


class LabelConstraint(Constraint):
    def __init__(self, in_labels, target_labels, penalty, penalty_window=0):
        self.in_labels = copy.copy(in_labels)
        self.out_labels = target_labels
        self.penalty = penalty
        self.window = penalty_window

    def apply(self, transition_cost, penalty, song, beat_names):
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

        return transition_cost, penalty, beat_names

    def __repr__(self):
        return "LabelConstraint"  


class GenericTimeSensitivePenalty(Constraint):
    def __init__(self, penalty):
        self.penalty = penalty

    def apply(self, transition_cost, penalty, song, beat_names):
        penalty[:n_beats, :] += self.penalty
        return transition_cost, penalty, beat_names


class EnergyConstraint(Constraint):
    # does not work with music duration constraint yet
    def __init__(self, penalty=3.0):
        self.penalty = penalty

    def apply(self, transition_cost, penalty, song, beat_names):
        sr = song.samplerate
        frames = song.all_as_mono()
        n_beats = len(song.analysis["beats"])

        energies = np.zeros(n_beats)
        for i, beat in enumerate(beat_names[:n_beats - 1]):
            start_frame = sr * beat
            end_frame = sr * beat_names[:n_beats][i + 1]
            beat_frames = frames[start_frame:end_frame]
            beat_frames *= np.hamming(len(beat_frames))
            energies[i] = np.sqrt(np.mean(beat_frames * beat_frames))

        energies[-1] = energies[-2]
        energies = [[x] for x in energies]

        dist_matrix = 10 * scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(energies, 'euclidean'))

        # shift it
        dist_matrix[:-1, :] = dist_matrix[1:, :]
        dist_matrix[-1, :] = np.inf

        transition_cost[:n_beats, :n_beats] *= (dist_matrix * self.penalty + 1)

        return transition_cost, penalty, beat_names

    def __repr__(self):
        return "EnergyConstraint: penalty=%f" % self.penalty


class PauseConstraint(Constraint):
    def __init__(self, min_length, max_length):
        self.min_len = min_length
        self.max_len = max_length
        # perhaps these costs should be based on the cost of a 
        # "bad" transition in the music.
        self.to_cost = 1.4
        # self.to_cost = 0.7
        # self.to_cost = 0.075
        self.bw_cost = 0.05


    def apply(self, transition_cost, penalty, song, beat_names):
        # we have to manage the pauses...
        n_beats = len(song.analysis["beats"])
        beat_len = song.analysis["avg_beat_duration"]
        min_beats = int(np.ceil(self.min_len / float(beat_len)))
        max_beats = int(np.floor(self.max_len / float(beat_len)))

        tc = self.to_cost
        # tc = self.to_cost * min_beats
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

        # add pauses to beat_names
        beat_names.extend(["p%d" % i for i in xrange(max_beats)])

        return new_trans, new_pen, beat_names

    def __repr__(self):
        return "PauseConstraint: min(%f), max(%f)" % (self.min_len, self.max_len)


class PauseEntryLabelChangeConstraint(Constraint):
    def __init__(self, target_labels, penalty_value):
        self.out_labels = target_labels
        self.p = penalty_value

    def apply(self, transition_cost, penalty, song, beat_names):
        n_beats = len(song.analysis["beats"])
        n_pauses = transition_cost.shape[0] - n_beats
        p0 = n_beats
        
        if n_pauses > 0:
            target_changes = [0]
            for l in xrange(1, len(self.out_labels)):
                target = self.out_labels[l]
                prev_target = self.out_labels[l - 1]
                if target != prev_target:
                    target_changes.append(l)
                    # target_changes.append(max(l - 4, 0))

            target_changes = np.array(target_changes)            

            penalty[p0, :] += self.p
            penalty[p0, target_changes] -= self.p

        return transition_cost, penalty, beat_names

    def __repr__(self):
        return "PauseEntryLabelChangeConstraint: penalty(%f)" % self.p


class PauseExitLabelChangeConstraint(Constraint):
    def __init__(self, target_labels, penalty_value):
        self.out_labels = target_labels
        self.p = penalty_value

    def apply(self, transition_cost, penalty, song, beat_names):
        n_beats = len(song.analysis["beats"])
        if transition_cost.shape[0] > n_beats:
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

        return transition_cost, penalty, beat_names

    def __repr__(self):
        return "PauseExitLabelChangeConstraint: penalty(%f)" % self.p


class NoveltyConstraint(Constraint):
    def __init__(self, in_labels, target_labels, penalty):
        self.in_labels = in_labels
        self.out_labels = target_labels
        self.penalty = penalty

    def apply(self, transition_cost, penalty, song, beat_names):
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

        return transition_cost, penalty, beat_names

    def __repr__(self):
        return "NoveltyConstraint"


class MusicDurationConstraint(Constraint):
    def __init__(self, min_length, max_length):
        self.minlen = min_length
        self.maxlen = max_length

    def apply(self, transition_cost, penalty, song, beat_names):
        beat_len = song.analysis["avg_beat_duration"]
        minlen = int(self.minlen / beat_len)
        maxlen = int(self.maxlen / beat_len)
        beats = song.analysis["beats"]
        n_beats = len(beats)
        n_pause_beats = transition_cost.shape[0] - n_beats

        # basically infinity.
        pen_val = 99999999.0

        # Create new transition cost table
        # (beat * beat index in max span) x (beat * beat index in max span of music)
        # Is this too large?
        new_tc_size = n_beats * maxlen + n_pause_beats
        p0 = n_beats * maxlen
        new_tc = np.empty((new_tc_size, new_tc_size))

        # tile the tc over this new table
        new_tc[:p0, :p0] = np.tile(transition_cost[:n_beats, :n_beats],
            (maxlen, maxlen))
        # tile the pause information as well
        new_tc[:p0, p0:] = np.tile(transition_cost[:n_beats, n_beats:],
            (maxlen, 1))
        new_tc[p0:, :p0] = np.tile(transition_cost[n_beats:, :n_beats],
            (1, maxlen))
        new_tc[p0:, p0:] = transition_cost[n_beats:, n_beats:]

        # Create new penalty table
        # (beat * beat index in max span) x (beats in output)
        new_pen = np.empty((new_tc_size, penalty.shape[1]))

        # tile the tc over this new table
        new_pen[:p0, :] = np.tile(penalty[:n_beats, :],
            (maxlen, 1))
        new_pen[p0:, :] = penalty[n_beats:, :]


        #--- CONSTRAINTS ---#
        # * don't start song in segment beat other than first
        new_pen[n_beats:(n_beats * maxlen), 0] += pen_val

        # * don't go to pause before minimum length music segment
        new_tc[:(n_beats * minlen), p0] += pen_val

        # * must go to pause if we're at the maxlen-th beat
        new_tc[n_beats * (maxlen - 1):n_beats * maxlen, :p0] += pen_val

        # * after pause, don't go to non-first segment beat
        new_tc[p0:, n_beats:p0] += pen_val

        # * don't move between beats that don't follow 
        #   the segment index
        new_tc[:p0, :p0] += pen_val
        for i in xrange(1, maxlen):
            new_tc[(i - 1) * n_beats:i * n_beats,
                   i * n_beats:(i + 1) * n_beats] -= pen_val

        # update beat_names
        pause_names = beat_names[n_beats:]
        new_beat_names = []
        for rep in xrange(maxlen):
            new_beat_names.extend(beat_names[:n_beats])
        new_beat_names.extend(pause_names)

        return new_tc, new_pen, new_beat_names

    def __repr__(self):
        return "MusicDurationConstraint"


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

