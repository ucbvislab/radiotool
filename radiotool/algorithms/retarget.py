from __future__ import print_function
import copy
from collections import namedtuple
import time
import logging

import numpy as np
import scipy.linalg

from ..composer import Composition, Segment, Volume, Label, RawVolume, Track
from novelty import novelty
from . import build_table_full_backtrace
from . import constraints as rt_constraints

Spring = namedtuple('Spring', ['time', 'duration'])
BEAT_DUR_KEY = "med_beat_duration"


class ArgumentException(Exception):
    pass


def retarget_to_length(song, duration, start=True, end=True, slack=5,
                       beats_per_measure=None):
    """Create a composition of a song that changes its length
    to a given duration.

    :param song: Song to retarget
    :type song: :py:class:`radiotool.composer.Song`
    :param duration: Duration of retargeted song (in seconds)
    :type duration: float
    :param start: Start the retargeted song at the
                  beginning of the original song
    :type start: boolean
    :param end: End the retargeted song at the end of the original song
    :type end: boolean
    :param slack: Track will be within slack seconds of the target
                  duration (more slack allows for better-sounding music)
    :type slack: float
    :returns: Composition of retargeted song
    :rtype: :py:class:`radiotool.composer.Composition`
    """

    duration = float(duration)

    constraints = [
        rt_constraints.TimbrePitchConstraint(
            context=0, timbre_weight=1.0, chroma_weight=1.0),
        rt_constraints.EnergyConstraint(penalty=.5),
        rt_constraints.MinimumLoopConstraint(8),
    ]

    if beats_per_measure is not None:
        constraints.append(
            rt_constraints.RhythmConstraint(beats_per_measure, .125))

    if start:
        constraints.append(
            rt_constraints.StartAtStartConstraint(padding=0))

    if end:
        constraints.append(
            rt_constraints.EndAtEndConstraint(padding=slack))

    comp, info = retarget(
        [song], duration, constraints=[constraints],
        fade_in_len=None, fade_out_len=None)

    # force the new track to extend to the end of the song
    if end == "end":
        last_seg = sorted(
            comp.segments,
            key=lambda seg:
            seg.comp_location_in_seconds + seg.duration_in_seconds
        )[-1]
        last_seg.duration_in_seconds = (
            song.duration_in_seconds - last_seg.start_in_seconds)

    path_cost = info["path_cost"]
    total_nonzero_cost = []
    total_nonzero_points = []
    for node in path_cost:
        if float(node.name) > 0.0:
            total_nonzero_cost.append(float(node.name))
            total_nonzero_points.append(float(node.time))

    transitions = zip(total_nonzero_points, total_nonzero_cost)

    for transition in transitions:
        comp.add_label(Label("crossfade", transition[0]))
    return comp


def retarget_with_change_points(song, cp_times, duration):
    """Create a composition of a song of a given duration that reaches
    music change points at specified times. This is still under
    construction. It might not work as well with more than
    2 ``cp_times`` at the moment.

    Here's an example of retargeting music to be 40 seconds long and
    hit a change point at the 10 and 30 second marks::

        song = Song("instrumental_music.wav")
        composition, change_points =\
            retarget.retarget_with_change_points(song, [10, 30], 40)
        composition.export(filename="retargeted_instrumental_music.")

    :param song: Song to retarget
    :type song: :py:class:`radiotool.composer.Song`
    :param cp_times: Times to reach change points (in seconds)
    :type cp_times: list of floats
    :param duration: Target length of retargeted music (in seconds)
    :type duration: float
    :returns: Composition of retargeted song and list of locations of
        change points in the retargeted composition
    :rtype: (:py:class:`radiotool.composer.Composition`, list)
    """
    analysis = song.analysis

    beat_length = analysis[BEAT_DUR_KEY]
    beats = np.array(analysis["beats"])

    # find change points
    cps = np.array(novelty(song, nchangepoints=4))
    cp_times = np.array(cp_times)

    # mark change points in original music
    def music_labels(t):
        # find beat closest to t
        closest_beat_idx = np.argmin(np.abs(beats - t))
        closest_beat = beats[closest_beat_idx]
        closest_cp = cps[np.argmin(np.abs(cps - closest_beat))]

        if np.argmin(np.abs(beats - closest_cp)) == closest_beat_idx:
            return "cp"
        else:
            return "noncp"

    # mark where we want change points in the output music
    # (a few beats of slack to improve the quality of the end result)
    def out_labels(t):
        if np.min(np.abs(cp_times - t)) < 1.5 * beat_length:
            return "cp"
        return "noncp"

    m_labels = [music_labels(i) for i in
                np.arange(0, song.duration_in_seconds, beat_length)]
    o_labels = [out_labels(i) for i in np.arange(0, duration, beat_length)]

    constraints = [
        rt_constraints.TimbrePitchConstraint(
            context=0, timbre_weight=1.0, chroma_weight=1.0),
        rt_constraints.EnergyConstraint(penalty=.5),
        rt_constraints.MinimumLoopConstraint(8),
        rt_constraints.NoveltyConstraint(m_labels, o_labels, 1.0)
    ]

    comp, info = retarget(
        [song], duration, constraints=[constraints],
        fade_in_len=None, fade_out_len=None)

    final_cp_locations = [beat_length * i
                          for i, label in enumerate(info['result_labels'])
                          if label == 'cp']

    return comp, final_cp_locations


def retarget(songs, duration, music_labels=None, out_labels=None,
             out_penalty=None, volume=None, volume_breakpoints=None,
             springs=None, constraints=None,
             min_beats=None, max_beats=None,
             fade_in_len=3.0, fade_out_len=5.0,
             **kwargs):
    """Retarget a song to a duration given input and output labels on
    the music.

    Suppose you like one section of a song, say, the guitar solo, and
    you want to create a three minute long version of the solo.
    Suppose the guitar solo occurs from the 150 second mark to the 200
    second mark in the original song.

    You can set the label the guitar solo with 'solo' and the rest of
    the song with 'other' by crafting the ``music_labels`` input
    function. And you can set the ``out_labels`` function to give you
    nothing but solo::

        def labels(t):
            if 150 < t < 200:
                return 'solo'
            return 'other'

        def target(t): return 'solo'

        song = Song("sweet-rock-song.wav")

        composition, info = retarget(song, 180,
            music_labels=labels, out_labels=target)

        composition.export(filename="super-long-solo")

    You can achieve much more complicated retargetings by adjusting
    the ``music_labels``, `out_labels` and ``out_penalty`` functions,
    but this should give you a basic sense of how to use the
    ``retarget`` function.

    :param song: Song to retarget
    :type song: :py:class:`radiotool.composer.Song`
    :param duration: Duration of retargeted song (in seconds)
    :type duration: float
    :param music_labels: A function that takes a time (in seconds) and
        returns the label (str) of the input music at that time
    :type music_labels: function
    :param out_labels: A function that takes a time (in seconds) and
        returns the desired label (str) of the output music at that
        time
    :type out_labels: function
    :param out_penalty: A function that takes a time (in seconds) and
        returns the penalty for not matching the correct output label
        at that time (default is 1.0)
    :type out_penalty: function
    :returns: Composition of retargeted song, and dictionary of
        information about the retargeting
    :rtype: (:py:class:`radiotool.composer.Composition`, dict)
    """

    # get song analysis
    if isinstance(songs, Track):
        songs = [songs]
    multi_songs = len(songs) > 1

    analyses = [s.analysis for s in songs]

    # generate labels for every beat in the input and output
    beat_lengths = [a[BEAT_DUR_KEY] for a in analyses]
    beats = [a["beats"] for a in analyses]

    beat_length = np.mean(beat_lengths)
    logging.info("Beat lengths of songs: {} (mean: {})".
                 format(beat_lengths, beat_length))

    if out_labels is not None:
        target = [out_labels(i) for i in np.arange(0, duration, beat_length)]
    else:
        target = ["" for i in np.arange(0, duration, beat_length)]

    if music_labels is not None:
        if not multi_songs:
            music_labels = [music_labels]
            music_labels = [item for sublist in music_labels
                            for item in sublist]
        if len(music_labels) != len(songs):
            raise ArgumentException("Did not specify {} sets of music labels".
                                    format(len(songs)))
        start = [[music_labels[i](j) for j in b] for i, b in enumerate(beats)]
    else:
        start = [["" for i in b] for b in beats]

    if out_penalty is not None:
        pen = np.array([out_penalty(i) for i in np.arange(
            0, duration, beat_length)])
    else:
        pen = np.array([1 for i in np.arange(0, duration, beat_length)])

    # we're using a valence/arousal constraint, so we need these
    in_vas = kwargs.pop('music_va', None)
    if in_vas is not None:
        if not multi_songs:
            in_vas = [in_vas]
            in_vas = [item for sublist in in_vas for item in sublist]
        if len(in_vas) != len(songs):
            raise ArgumentException("Did not specify {} sets of v/a labels".
                                    format(len(songs)))
        for i, in_va in enumerate(in_vas):
            if callable(in_va):
                in_va = np.array([in_va(j) for j in beats[i]])
            in_vas[i] = in_va

    target_va = kwargs.pop('out_va', None)
    if callable(target_va):
        target_va = np.array(
            [target_va(i) for i in np.arange(0, duration, beat_length)])

    # set constraints
    if constraints is None:
        min_pause_len = 20.
        max_pause_len = 35.
        min_pause_beats = int(np.ceil(min_pause_len / beat_length))
        max_pause_beats = int(np.floor(max_pause_len / beat_length))

        constraints = [(
            rt_constraints.PauseConstraint(
                min_pause_beats, max_pause_beats,
                to_penalty=1.4, between_penalty=.05, unit="beats"),
            rt_constraints.PauseEntryVAChangeConstraint(target_va, .005),
            rt_constraints.PauseExitVAChangeConstraint(target_va, .005),
            rt_constraints.TimbrePitchConstraint(
                context=0, timbre_weight=1.5, chroma_weight=1.5),
            rt_constraints.EnergyConstraint(penalty=0.5),
            rt_constraints.MinimumLoopConstraint(8),
            rt_constraints.ValenceArousalConstraint(
                in_va, target_va, pen * .125),
            rt_constraints.NoveltyVAConstraint(in_va, target_va, pen),
        ) for in_va in in_vas]
    else:
        max_pause_beats = 0
        if len(constraints) > 0:
            if isinstance(constraints[0], rt_constraints.Constraint):
                constraints = [constraints]

    pipelines = [rt_constraints.ConstraintPipeline(constraints=c_set)
                 for c_set in constraints]

    trans_costs = []
    penalties = []
    all_beat_names = []

    for i, song in enumerate(songs):
        (trans_cost, penalty, bn) = pipelines[i].apply(song, len(target))
        trans_costs.append(trans_cost)
        penalties.append(penalty)
        all_beat_names.append(bn)

    logging.info("Combining tables")
    total_music_beats = int(np.sum([len(b) for b in beats]))
    total_beats = total_music_beats + max_pause_beats

    # combine transition cost tables

    trans_cost = np.ones((total_beats, total_beats)) * np.inf
    sizes = [len(b) for b in beats]
    idx = 0
    for i, size in enumerate(sizes):
        trans_cost[idx:idx + size, idx:idx + size] =\
            trans_costs[i][:size, :size]
        idx += size

    trans_cost[:total_music_beats, total_music_beats:] =\
        np.vstack([tc[:len(beats[i]), len(beats[i]):]
                   for i, tc in enumerate(trans_costs)])

    trans_cost[total_music_beats:, :total_music_beats] =\
        np.hstack([tc[len(beats[i]):, :len(beats[i])]
                  for i, tc in enumerate(trans_costs)])

    trans_cost[total_music_beats:, total_music_beats:] =\
        trans_costs[0][len(beats[0]):, len(beats[0]):]

    # combine penalty tables
    penalty = np.empty((total_beats, penalties[0].shape[1]))

    penalty[:total_music_beats, :] =\
        np.vstack([p[:len(beats[i]), :] for i, p in enumerate(penalties)])

    penalty[total_music_beats:, :] = penalties[0][len(beats[0]):, :]

    logging.info("Building cost table")

    # compute the dynamic programming table (prev python method)
    # cost, prev_node = _build_table(analysis, duration, start, target, pen)

    # first_pause = 0
    # if max_pause_beats > 0:
    first_pause = total_music_beats

    if min_beats is None:
        min_beats = 0
    elif min_beats is 'default':
        min_beats = int(20. / beat_length)

    if max_beats is None:
        max_beats = -1
    elif max_beats is 'default':
        max_beats = int(90. / beat_length)
        max_beats = min(max_beats, penalty.shape[1])

    tc2 = np.nan_to_num(trans_cost)
    pen2 = np.nan_to_num(penalty)

    beat_names = []
    for i, bn in enumerate(all_beat_names):
        for b in bn:
            if not str(b).startswith('p'):
                beat_names.append((i, float(b)))
    beat_names.extend([('p', i) for i in xrange(max_pause_beats)])

    result_labels = []

    logging.info("Running optimization (full backtrace, memory efficient)")
    logging.info("\twith min_beats(%d) and max_beats(%d) and first_pause(%d)" %
                 (min_beats, max_beats, first_pause))

    song_starts = [0]
    for song in songs:
        song_starts.append(song_starts[-1] + len(song.analysis["beats"]))
    song_ends = np.array(song_starts[1:], dtype=np.int32)
    song_starts = np.array(song_starts[:-1], dtype=np.int32)

    t1 = time.clock()
    path_i, path_cost = build_table_full_backtrace(
        tc2, pen2, song_starts, song_ends,
        first_pause=first_pause, max_beats=max_beats, min_beats=min_beats)
    t2 = time.clock()
    logging.info("Built table (full backtrace) in {} seconds"
                 .format(t2 - t1))

    path = []
    if max_beats == -1:
        max_beats = min_beats + 1

    first_pause_full = max_beats * first_pause
    n_beats = first_pause
    for i in path_i:
        if i >= first_pause_full:
            path.append(('p', i - first_pause_full))
            result_labels.append(None)
            # path.append('p' + str(i - first_pause_full))
        else:
            path.append(beat_names[i % n_beats])
            song_i = path[-1][0]
            beat_name = path[-1][1]
            result_labels.append(
                start[song_i][np.where(np.array(beats[song_i]) ==
                              beat_name)[0][0]])
            # path.append(float(beat_names[i % n_beats]))

    # else:
    #     print("Running optimization (fast, full table)")
    #     # this won't work right now- needs to be updated
    #     # with the multi-song approach

    #     # fortran method
    #     t1 = time.clock()
    #     cost, prev_node = build_table(tc2, pen2)
    #     t2 = time.clock()
    #     print("Built table (fortran) in {} seconds".format(t2 - t1))
    #     res = cost[:, -1]
    #     best_idx = N.argmin(res)
    #     if N.isfinite(res[best_idx]):
    #         path, path_cost, path_i = _reconstruct_path(
    #             prev_node, cost, beat_names, best_idx, N.shape(cost)[1] - 1)
    #         # path_i = [beat_names.index(x) for x in path]
    #     else:
    #         # throw an exception here?
    #         return None

    #     path = []
    #     result_labels = []
    #     if max_pause_beats == 0:
    #         n_beats = total_music_beats
    #         first_pause = n_beats
    #     else:
    #         n_beats = first_pause
    #     for i in path_i:
    #         if i >= first_pause:
    #             path.append(('p', i - first_pause))
    #             result_labels.append(None)
    #         else:
    #             path.append(beat_names[i % n_beats])
    #             song_i = path[-1][0]
    #             beat_name = path[-1][1]
    #             result_labels.append(
    #                 start[song_i][N.where(N.array(beats[song_i]) ==
    #                               beat_name)[0][0]])

    # return a radiotool Composition
    logging.info("Generating audio")
    (comp, cf_locations, result_full_labels,
     cost_labels, contracted, result_volume) =\
        _generate_audio(
            songs, beats, path, path_cost, start,
            volume=volume,
            volume_breakpoints=volume_breakpoints,
            springs=springs,
            fade_in_len=fade_in_len, fade_out_len=fade_out_len)

    info = {
        "beat_length": beat_length,
        "contracted": contracted,
        "cost": np.sum(path_cost) / len(path),
        "path": path,
        "path_i": path_i,
        "target_labels": target,
        "result_labels": result_labels,
        "result_full_labels": result_full_labels,
        "result_volume": result_volume,
        "transitions": [Label("crossfade", loc) for loc in cf_locations],
        "path_cost": cost_labels
    }

    return comp, info


def _reconstruct_path(prev_node, cost_table, beat_names, end, length):
    path = []
    path.append(end)
    node = end
    while length > 0:
        node = prev_node[int(node), length]
        path.append(node)
        length -= 1

    beat_path = [beat_names[int(n)] for n in reversed(path)]

    path_cost = []
    prev_cost = 0.0
    for li, bi in enumerate(reversed(path)):
        this_cost = cost_table[bi, li]
        path_cost.append(this_cost - prev_cost)
        prev_cost = this_cost

    path_i = [int(x) for x in reversed(path)]

    return beat_path, path_cost, path_i


def _build_table_from_costs(trans_cost, penalty):
    # create cost matrix
    cost = np.zeros(penalty.shape)
    prev_node = np.zeros(penalty.shape)

    cost[:, 0] = penalty[:, 0]

    for l in xrange(1, penalty.shape[1]):
        tc = penalty[:, l] + trans_cost + cost[:, l - 1][:, np.newaxis]
        min_nodes = __fast_argmin_axis_0(tc)
        min_vals = np.amin(tc, axis=0)
        cost[:, l] = min_vals
        prev_node[:, l] = min_nodes

    return cost, prev_node


def _build_table(analysis, duration, start, target, out_penalty):
    beats = analysis["beats"]
    trans_cost = np.copy(analysis["dense_dist"])

    # shift it over
    trans_cost[:-1, :] = trans_cost[1:, :]
    trans_cost[-1, :] = np.inf

    # create cost matrix
    cost = np.empty((len(beats), len(target)))
    prev_node = np.empty((len(beats), len(target)))

    # set initial values for first row of the cost table
    first_target = target[0]
    init = [0] * len(start)
    for i, label in enumerate(start):
        if label == first_target or label is None or first_target is None:
            init[i] = 0.0
        else:
            init[i] = 1.0
    cost[:, 0] = init

    # no self-jumps
    np.fill_diagonal(trans_cost, np.inf)

    min_jump = 4

    # no jumps within min-jump
    if min_jump and min_jump > 0:
        total_len = np.shape(trans_cost)[0]
        for idx in range(total_len):
            for diag_idx in range(-(min_jump - 1), min_jump):
                if 0 < idx + diag_idx < total_len and diag_idx != 1:
                    trans_cost[idx, idx + diag_idx] = np.inf

    # create label penalty table
    penalty = np.ones((len(beats), len(target))) * np.array(out_penalty)

    penalty_window = 4
    for n_i in xrange(len(beats)):
        node_label = start[n_i]
        for l in xrange(1, len(target) - 1):
            prev_target = target[l - 1]
            next_target = target[l + 1]
            target_label = target[l]

            if node_label == target_label or target_label is None:
                penalty[n_i, l] = 0.0

            # if target_label != prev_target:
            #     # reduce penalty for beats prior
            #     span = min(penalty_window, l)
            #     P[n_i, l - span:l] = N.linspace(1.0, 0.0, num=span)

            # if target_label != next_target:
            #     # reduce penalty for beats later
            #     span = min(penalty_window, len(target) - l - 1)
            #     penalty[n_i, l + 1:l + span + 1] =\
            #         N.linspace(0.0, 1.0, num=span)

        # set penalty for the first and last targets
        for l in [0, len(target) - 1]:
            target_label = target[l]
            if node_label == target_label or target_label is None:
                penalty[n_i, l] = 0.0

    # building the remainder of the table
    for l in xrange(1, len(target)):
        tc = penalty[:, l] + trans_cost + cost[:, l - 1][:, np.newaxis]
        min_nodes = __fast_argmin_axis_0(tc)
        min_vals = np.amin(tc, axis=0)
        cost[:, l] = min_vals
        prev_node[:, l] = min_nodes

        # for n_i in xrange(len(beats)):
        #     total_cost =\
        #        penalty[n_i, l] + trans_cost[:, n_i] + cost[:, l - 1]
        #     min_node = N.argmin(total_cost)
        #     cost[n_i, l] = total_cost[min_node]
        #     prev_node[n_i, l] = min_node

    # result:
    return cost, prev_node


def __fast_argmin_axis_0(a):
    # http://stackoverflow.com/questions/17840661/
    #    is-there-a-way-to-make-numpy-argmin-as-fast-as-min
    matches = np.nonzero((a == np.min(a, axis=0)).ravel())[0]
    rows, cols = np.unravel_index(matches, a.shape)
    argmin_array = np.empty(a.shape[1], dtype=np.intp)
    argmin_array[cols] = rows
    return argmin_array


def _generate_audio(songs, beats, new_beats, new_beats_cost, music_labels,
                    volume=None, volume_breakpoints=None,
                    springs=None, fade_in_len=3.0, fade_out_len=5.0):
    # assuming same sample rate for all songs

    logging.info("Building volume")
    if volume is not None and volume_breakpoints is not None:
        raise Exception("volume and volume_breakpoints cannot both be defined")
    if volume_breakpoints is None:
        if volume is None:
            volume = 1.0
        volume_array = np.array([volume])

    if volume_breakpoints is not None:
        volume_array = volume_breakpoints.to_array(songs[0].samplerate)

    result_volume = np.zeros(volume_array.shape)

    min_channels = min([x.channels for x in songs])

    comp = Composition(channels=min_channels)

    # currently assuming no transitions between different songs

    beat_length = np.mean([song.analysis[BEAT_DUR_KEY]
                          for song in songs])

    audio_segments = []
    segment_song_indicies = [new_beats[0][0]]
    current_seg = [0, 0]
    if new_beats[0][0] == 'p':
        current_seg = 'p'

    for i, (song_i, b) in enumerate(new_beats):
        if segment_song_indicies[-1] != song_i:
            segment_song_indicies.append(song_i)

        if current_seg == 'p' and song_i != 'p':
            current_seg = [i, i]
        elif current_seg != 'p' and song_i == 'p':
            audio_segments.append(current_seg)
            current_seg = 'p'
        elif current_seg != 'p':
            current_seg[1] = i
    if current_seg != 'p':
        audio_segments.append(current_seg)

    segment_song_indicies = [x for x in segment_song_indicies if x != 'p']

    beats = [np.array(b) for b in beats]
    score_start = 0
    current_loc = 0.0
    last_segment_beat = 0

    comp.add_tracks(songs)

    all_cf_locations = []

    aseg_fade_ins = []

    logging.info("Building audio")
    for (aseg, song_i) in zip(audio_segments, segment_song_indicies):
        segments = []
        # TODO: is this +1 correct?
        starts = np.array([x[1] for x in new_beats[aseg[0]:aseg[1] + 1]])

        bis = [np.nonzero(beats[song_i] == b)[0][0] for b in starts]
        dists = np.zeros(len(starts))
        durs = np.zeros(len(starts))

        for i, beat in enumerate(starts):
            if i < len(bis) - 1:
                if bis[i] + 1 != bis[i + 1]:
                    dists[i + 1] = 1
            if bis[i] + 1 >= len(beats[song_i]):
                # use the average beat duration if we don't know
                # how long the beat is supposed to be
                logging.warning("USING AVG BEAT DURATION IN SYNTHESIS -\
                    POTENTIALLY NOT GOOD")
                durs[i] = songs[song_i].analysis[BEAT_DUR_KEY]
            else:
                durs[i] = beats[song_i][bis[i] + 1] - beats[song_i][bis[i]]

        # add pause duration to current location
        # current_loc +=\
            # (aseg[0] - last_segment_beat) *\
            #      song.analysis[BEAT_DUR_KEY]

        # catch up to the pause
        current_loc = max(
            aseg[0] * beat_length,
            current_loc)

        last_segment_beat = aseg[1] + 1

        cf_durations = []
        seg_start = starts[0]
        seg_start_loc = current_loc

        cf_locations = []

        segment_starts = [0]
        try:
            segment_starts.extend(np.where(dists == 1)[0])
        except:
            pass

        # print "segment starts", segment_starts

        for i, s_i in enumerate(segment_starts):
            if i == len(segment_starts) - 1:
                # last segment?
                seg_duration = np.sum(durs[s_i:])
            else:
                next_s_i = segment_starts[i + 1]
                seg_duration = np.sum(durs[s_i:next_s_i])

                cf_durations.append(durs[next_s_i])
                cf_locations.append(current_loc + seg_duration)

            seg_music_location = starts[s_i]

            seg = Segment(songs[song_i], current_loc,
                          seg_music_location, seg_duration)

            segments.append(seg)

            # update location for next segment
            current_loc += seg_duration

        # for i, start in enumerate(starts):
        #     dur = durs[i]
        #     current_loc += dur
        #     if i == 0 or dists[i - 1] == 0:
        #         pass
        #         # dur = durs[i]
        #         # current_loc += dur
        #     else:
        #         seg = Segment(song, seg_start_loc, seg_start,
        #                       current_loc - seg_start_loc)
        #         print "segment duration", current_loc - seg_start_loc
        #         segments.append(seg)

        #         # track = Track(wav_fn, t["name"])
        #         # comp.add_track(track)
        #         # dur = durs[i]
        #         cf_durations.append(dur)
        #         cf_locations.append(current_loc)

        #         seg_start_loc = current_loc
        #         seg_start = start

        #         # current_loc += dur

        # last_seg = Segment(song, seg_start_loc, seg_start,
        #     current_loc - seg_start_loc)
        # segments.append(last_seg)

        comp.add_segments(segments)

        if segments[-1].comp_location + segments[-1].duration >\
                len(volume_array):

            diff = len(volume_array) -\
                (segments[-1].comp_location + segments[-1].duration)
            new_volume_array =\
                np.ones(segments[-1].comp_location + segments[-1].duration) *\
                volume_array[-1]
            new_volume_array[:len(volume_array)] = volume_array
            volume_array = new_volume_array
            result_volume = np.zeros(new_volume_array.shape)

        for i, seg in enumerate(segments[:-1]):
            logging.info(cf_durations[i], seg.duration_in_seconds,
                         segments[i + 1].duration_in_seconds)
            rawseg = comp.cross_fade(seg, segments[i + 1], cf_durations[i])

            # decrease volume along crossfades
            volume_frames = volume_array[
                rawseg.comp_location:rawseg.comp_location + rawseg.duration]
            raw_vol = RawVolume(rawseg, volume_frames)
            comp.add_dynamic(raw_vol)

            result_volume[rawseg.comp_location:
                          rawseg.comp_location + rawseg.duration] =\
                volume_frames

        s0 = segments[0]
        sn = segments[-1]

        if fade_in_len is not None:
            fi_len = min(fade_in_len, s0.duration_in_seconds)
            fade_in_len_samps = fi_len * s0.track.samplerate
            fade_in = comp.fade_in(s0, fi_len, fade_type="linear")
            aseg_fade_ins.append(fade_in)
        else:
            fade_in = None

        if fade_out_len is not None:
            fo_len = min(5.0, sn.duration_in_seconds)
            fade_out_len_samps = fo_len * sn.track.samplerate
            fade_out = comp.fade_out(sn, fade_out_len, fade_type="exponential")
        else:
            fade_out = None

        prev_end = 0.0

        for seg in segments:
            volume_frames = volume_array[
                seg.comp_location:seg.comp_location + seg.duration]

            # this can happen on the final segment:
            if len(volume_frames) == 0:
                volume_frames = np.array([prev_end] * seg.duration)
            elif len(volume_frames) < seg.duration:
                delta = [volume_frames[-1]] *\
                    (seg.duration - len(volume_frames))
                volume_frames = np.r_[volume_frames, delta]
            raw_vol = RawVolume(seg, volume_frames)
            comp.add_dynamic(raw_vol)

            try:
                result_volume[seg.comp_location:
                              seg.comp_location + seg.duration] = volume_frames
            except ValueError:
                diff = (seg.comp_location + seg.duration) - len(result_volume)
                result_volume = np.r_[result_volume, np.zeros(diff)]
                result_volume[seg.comp_location:
                              seg.comp_location + seg.duration] = volume_frames

            if len(volume_frames) != 0:
                prev_end = volume_frames[-1]

            # vol = Volume.from_segment(seg, volume)
            # comp.add_dynamic(vol)

        if fade_in is not None:
            result_volume[s0.comp_location:
                          s0.comp_location + fade_in_len_samps] *=\
                fade_in.to_array(channels=1).flatten()
        if fade_out is not None:
            result_volume[sn.comp_location + sn.duration - fade_out_len_samps:
                          sn.comp_location + sn.duration] *=\
                fade_out.to_array(channels=1).flatten()

        all_cf_locations.extend(cf_locations)

    # result labels
    label_time = 0.0
    pause_len = beat_length
    # pause_len = song.analysis[BEAT_DUR_KEY]
    result_full_labels = []
    prev_label = -1
    for beat_i, (song_i, beat) in enumerate(new_beats):
        if song_i == 'p':
            current_label = None
            if current_label != prev_label:
                result_full_labels.append(Label("pause", label_time))
            prev_label = None

            # label_time += pause_len
            # catch up
            label_time = max(
                (beat_i + 1) * pause_len,
                label_time)
        else:
            beat_i = np.where(np.array(beats[song_i]) == beat)[0][0]
            next_i = beat_i + 1
            current_label = music_labels[song_i][beat_i]
            if current_label != prev_label:
                if current_label is None:
                    result_full_labels.append(Label("none", label_time))
                else:
                    result_full_labels.append(Label(current_label, label_time))
            prev_label = current_label

            if (next_i >= len(beats[song_i])):
                logging.warning("USING AVG BEAT DURATION - "
                                "POTENTIALLY NOT GOOD")
                label_time += songs[song_i].analysis[BEAT_DUR_KEY]
            else:
                label_time += beats[song_i][next_i] - beat

    # result costs
    cost_time = 0.0
    result_cost = []
    for i, (song_i, b) in enumerate(new_beats):
        result_cost.append(Label(new_beats_cost[i], cost_time))

        if song_i == 'p':
            # cost_time += pause_len
            # catch up
            cost_time = max(
                (i + 1) * pause_len,
                cost_time)
        else:
            beat_i = np.where(np.array(beats[song_i]) == b)[0][0]
            next_i = beat_i + 1

            if (next_i >= len(beats[song_i])):
                cost_time += songs[song_i].analysis[BEAT_DUR_KEY]
            else:
                cost_time += beats[song_i][next_i] - b

    logging.info("Contracting pause springs")
    contracted = []
    min_contraction = 0.5
    if springs is not None:
        offset = 0.0
        for spring in springs:
            contracted_time, contracted_dur = comp.contract(
                spring.time - offset, spring.duration,
                min_contraction=min_contraction)
            if contracted_dur > 0:
                logging.info("Contracted", contracted_time,
                             "at", contracted_dur)

                # move all the volume frames back
                c_time_samps = contracted_time * segments[0].track.samplerate
                c_dur_samps = contracted_dur * segments[0].track.samplerate
                result_volume = np.r_[
                    result_volume[:c_time_samps],
                    result_volume[c_time_samps + c_dur_samps:]]

                # can't move anything EARLIER than contracted_time

                new_cf = []
                for cf in all_cf_locations:
                    if cf > contracted_time:
                        new_cf.append(
                            max(cf - contracted_dur, contracted_time))
                    else:
                        new_cf.append(cf)
                all_cf_locations = new_cf

                # for lab in result_full_labels:
                #     if lab.time > contracted_time + contracted_dur:
                #         lab.time -= contracted_dur

                first_label = True
                for lab_i, lab in enumerate(result_full_labels):
                    # is this contracted in a pause that already started?
                    # if lab_i + 1 < len(result_full_labels):
                    #     next_lab = result_full_labels[lab_i + 1]
                    #     if lab.time < contracted_time <= next_lab.time:
                    #         first_label = False

                    # if lab.time > contracted_time:
                    #     # TODO: fix this hack
                    #     if lab.name == "pause" and first_label:
                    #         pass
                    #     else:
                    #         lab.time -= contracted_dur
                    #     first_label = False

                    try:
                        if lab.time == contracted_time and\
                            result_full_labels[lab_i + 1].time -\
                                contracted_dur == lab.time:

                            logging.warning("LABEL HAS ZERO LENGTH", lab)
                    except:
                        pass

                    if lab.time > contracted_time:
                        logging.info("\tcontracting label", lab)
                        lab.time = max(
                            lab.time - contracted_dur, contracted_time)
                        # lab.time -= contracted_dur
                        logging.info("\t\tto", lab)

                new_result_cost = []
                for cost_lab in result_cost:
                    if cost_lab.time <= contracted_time:
                        # cost is before contracted time
                        new_result_cost.append(cost_lab)
                    elif contracted_time < cost_lab.time <=\
                            contracted_time + contracted_dur:
                        # cost is during contracted time
                        # remove these labels
                        if cost_lab.name > 0:
                            logging.warning("DELETING nonzero cost label",
                                            cost_lab.name, cost_lab.time)
                    else:
                        # cost is after contracted time
                        cost_lab.time = max(
                            cost_lab.time - contracted_dur, contracted_time)
                        # cost_lab.time -= contracted_dur
                        new_result_cost.append(cost_lab)

                # new_result_cost = []
                # first_label = True
                # # TODO: also this hack. bleh.
                # for cost_lab in result_cost:
                #     if cost_lab.time < contracted_time:
                #         new_result_cost.append(cost_lab)
                #     elif cost_lab.time > contracted_time and\
                #             cost_lab.time <= contracted_time +\
                #                contracted_dur:
                #         if first_label:
                #             cost_lab.time = contracted_time
                #             new_result_cost.append(cost_lab)
                #         elif cost_lab.name > 0:
                #             print "DELETING nonzero cost label:",\
                #                 cost_lab.name, cost_lab.time
                #         first_label = False
                #     elif cost_lab.time > contracted_time + contracted_dur:
                #         cost_lab.time -= contracted_dur
                #         new_result_cost.append(cost_lab)
                #         first_label = False
                result_cost = new_result_cost

                contracted.append(
                    Spring(contracted_time + offset, contracted_dur))
                offset += contracted_dur

    for fade in aseg_fade_ins:
        for spring in contracted:
            if (spring.time - 1 <
                    fade.comp_location_in_seconds <
                    spring.time + spring.duration + 1):

                result_volume[
                    fade.comp_location:
                    fade.comp_location + fade.duration] /=\
                    fade.to_array(channels=1).flatten()

                fade.fade_type = "linear"
                fade.duration_in_seconds = 2.0
                result_volume[
                    fade.comp_location:
                    fade.comp_location + fade.duration] *=\
                    fade.to_array(channels=1).flatten()

                logging.info("Changing fade at {}".format(
                    fade.comp_location_in_seconds))

    # for seg in comp.segments:
    #     print seg.comp_location, seg.duration
    # print
    # for dyn in comp.dynamics:
    #     print dyn.comp_location, dyn.duration

    # add all the segments to the composition
    # comp.add_segments(segments)

    # all_segs = []

    # for i, seg in enumerate(segments[:-1]):
    #     rawseg = comp.cross_fade(seg, segments[i + 1], cf_durations[i])
    #     all_segs.extend([seg, rawseg])

    #     # decrease volume along crossfades
    #     rawseg.track.frames *= music_volume

    # all_segs.append(segments[-1])

    # add dynamic for music
    # vol = Volume(song, 0.0,
    #     (last_seg.comp_location + last_seg.duration) /
    #        float(song.samplerate),
    #     volume)
    # comp.add_dynamic(vol)

    # cf durs?
    # durs

    return (comp, all_cf_locations, result_full_labels,
            result_cost, contracted, result_volume)
