
import numpy as N

from ..composer import Composition, Segment, Volume
from novelty import novelty


def retarget_to_length(song, duration, start=True, end=True):

    if not start and not end:
        comp = retarget(song, duration)
    else:
        analysis = song.analysis
        beats = analysis["beats"]
        beat_length = analysis["avg_beat_duration"]
        def music_labels(t):
            if t <= beats[0] and start:
                return "start"
            elif t >= beats[-1] and end:
                return "end"
            else:
                return ""
        def out_labels(t):
            if t == 0 and start:
                return "start"
            elif t == N.arange(0, duration, beat_length)[-1] and end:
                return "end"
            else:
                return ""
        def out_penalty(t):
            if t == 0 and start:
                return N.inf
            elif t == N.arange(0, duration, beat_length)[-1] and end:
                return N.inf
            else:
                return 1

        comp = retarget(song, duration, music_labels, out_labels, out_penalty)

        # and the beatless ending to the composition
        last_seg = sorted(comp.segments, key=lambda k: k.comp_location + k.duration)[-1]

        seg = Segment(song, comp.duration / float(song.samplerate),
            (last_seg.start + last_seg.duration) / float(song.samplerate),
            (song.duration - last_seg.start - last_seg.duration) / float(song.samplerate))
        comp.add_segment(seg)

    comp.export(
        adjust_dynamics=False,
        filename="retarget_length_test",
        channels=1,
        filetype='wav',
        samplerate=44100,
        separate_tracks=False
    )

    return comp


def retarget_with_change_points(song, cp_times, duration):
    analysis = song.analysis
    beat_length = analysis["avg_beat_duration"]

    # find change points
    cps = N.array(novelty(song, nchangepoints=1))

    def music_labels(t):
        if N.min(N.abs(cps - t)) < .5 * beat_length:
            return "cp"
        else:
            return "noncp"

    def out_labels(t):
        for cp_time in cp_times:
            if abs(cp_time - t) < .5 * beat_length:
                return "cp"
        return "noncp"

    comp = retarget(song, duration, music_labels, out_labels)

    comp.export(
        adjust_dynamics=False,
        filename="retarget_test",
        channels=1,
        filetype='wav',
        samplerate=44100,
        separate_tracks=False
    )

    return comp


def retarget(song, duration, music_labels=None, out_labels=None, out_penalty=None):
    # labels can be array, array of arrays, or function
    # for now, assume music_labels and out_labels are time functions

    # get song analysis
    analysis = song.analysis

    # generate labels for every beat in the input and output
    beat_length = analysis["avg_beat_duration"]
    beats = analysis["beats"]

    if out_labels is not None:
        target = [out_labels(i) for i in N.arange(0, duration, beat_length)]
    else:
        target = ["" for i in N.arange(0, duration, beat_length)]

    if music_labels is not None:
        start = [music_labels(i) for i in beats]
    else:
        start = ["" for i in beats]

    if out_penalty is not None:
        pen = [out_penalty(i) for i in N.arange(0, duration, beat_length)]
    else:
        pen = [1 for i in N.arange(0, duration, beat_length)]
    
    # compute the dynamic programming table
    cost, prev_node = _build_table(analysis, duration, start, target, pen)

    # find the cheapest path    
    res = cost[:, -1]
    best_idx = N.argmin(res)
    if N.isfinite(res[best_idx]):
        path = _reconstruct_path(
            prev_node, analysis["beats"], best_idx, N.shape(cost)[1] - 1)
    else:
        # throw an exception here?
        return None

    # how did we do?
    result_labels = [start[N.where(N.array(beats) == i)[0][0]] for i in path]

    matched = len(N.where(N.array(result_labels) == N.array(target))[0])

    print "Matched %d of %d labels" % (matched, len(target))

    # return a composition?
    comp = _generate_audio(song, beats, path)

    print "path", path

    return comp


def _reconstruct_path(prev_node, beats, end, length):
    path = []
    path.append(end)
    node = end
    while length > 0:
        node = prev_node[int(node), length]
        path.append(node)
        length -= 1
    return [beats[int(n)] for n in reversed(path)]


def _build_table(analysis, duration, start, target, out_penalty):
    beats = analysis["beats"]
    trans_cost = analysis["dense_dist"]

    # shift it over
    trans_cost[:-1, :] = trans_cost[1:, :]
    trans_cost[-1, :] = N.inf

    # create cost matrix
    cost = N.empty((len(beats), len(target)))
    prev_node = N.empty((len(beats), len(target)))

    # set initial values for first row of the cost table
    first_target = target[0]
    init = [0] * len(start)
    for i, label in enumerate(start):
        if label == first_target or label is None or first_target is None:
            init[i] = 0.0
        else:
            init[i] = 1.0
    cost[:, 0] = init

    # create label penalty table
    penalty = N.ones((len(beats), len(target))) * N.array(out_penalty)

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
            #     penalty[n_i, l + 1:l + span + 1] = N.linspace(0.0, 1.0, num=span)

        # set penalty for the first and last targets
        for l in [0, len(target) - 1]:
            target_label = target[l]
            if node_label == target_label or target_label is None:
                penalty[n_i, l] = 0.0


    # building the remainder of the table
    for l in xrange(1, len(target)):
        for n_i in xrange(len(beats)):
            total_cost = penalty[n_i, l] + trans_cost[:, n_i] + cost[:, l - 1]
            min_node = N.argmin(total_cost)
            cost[n_i, l] = total_cost[min_node]
            prev_node[n_i, l] = min_node

    # result:
    return cost, prev_node


def _generate_audio(song, beats, new_beats):
    comp = Composition(channels=1)

    # music_volume = .10
    music_volume = 1.0

    starts = N.array(new_beats)
    beats = N.array(beats)

    bis = [N.nonzero(beats == b)[0][0] for b in starts]
    dists = [0] * len(starts)
    durs = [0] * len(starts)

    for i, beat in enumerate(starts):
        if i < len(bis) - 1:
            if bis[i] + 1 != bis[i + 1]:
                dists[i] = 1
            if bis[i] + 1 >= len(beats):
                durs[i] = beats[bis[i]] - beats[bis[i] - 1]
            else:
                durs[i] = beats[bis[i] + 1] - beats[bis[i]]

    score_start = 0
    comp.add_track(song)
    current_loc = float(score_start)

    segments = []
    cf_durations = []
    seg_start = starts[0]
    seg_start_loc = current_loc

    for i, start in enumerate(starts):
        if i == 0 or dists[i - 1] == 0:
            dur = durs[i]
            current_loc += dur
        else:
            seg = Segment(song, seg_start_loc, seg_start,
                          current_loc - seg_start_loc)
            segments.append(seg)

            # track = Track(wav_fn, t["name"])
            # comp.add_track(track)
            dur = durs[i]
            cf_durations.append(dur)
            print "Crossfade at", current_loc

            seg_start_loc = current_loc
            seg_start = start

            current_loc += dur

    last_seg = Segment(song, seg_start_loc, seg_start,
        current_loc - seg_start_loc)
    segments.append(last_seg)

    comp.add_segments(segments)

    all_segs = []

    for i, seg in enumerate(segments[:-1]):
        rawseg = comp.cross_fade(seg, segments[i + 1], cf_durations[i])
        all_segs.extend([seg, rawseg])

        # decrease volume along crossfades
        rawseg.track.frames *= music_volume

    all_segs.append(segments[-1])

    # add dynamic for music
    volume = Volume(song, 0.0,
        (last_seg.comp_location + last_seg.duration) / 44100.,
        music_volume)
    comp.add_dynamic(volume)

    # cf durs?
    # durs

    return comp
