import numpy as N

from ..composer import Composition, Segment, Volume
from novelty import novelty


def retarget_to_length(song, duration, start=True, end=True, slack=5):
    """Create a composition of a song that changes its length
    to a given duration.

    :param song: Song to retarget
    :type song: :py:class:`radiotool.composer.Song`
    :param duration: Duration of retargeted song (in seconds)
    :type duration: float
    :param start: Start the retargeted song at the beginning of the original song
    :type start: boolean
    :param end: End the retargeted song at the end of the original song
    :type end: boolean
    :param slack: Track will be within slack seconds of the target duration (more slack allows for better-sounding music)
    :type slack: float
    :returns: Composition of retargeted song
    :rtype: :py:class:`radiotool.composer.Composition`
    """

    if not start and not end:
        comp, info = retarget(song, duration)
    else:
        analysis = song.analysis
        beats = analysis["beats"]
        beat_length = analysis["avg_beat_duration"]

        ending_length = song.duration_in_seconds - beats[-1]
        new_duration = duration - ending_length - slack
        slack_beats = int((2 * slack) / beat_length)

        def music_labels(t):
            if t <= beats[0] and start:
                return "start"
            elif t >= beats[-1 - slack_beats] and end:
                return "end"
            else:
                return ""
        def out_labels(t):
            if t == 0 and start:
                return "start"
            elif t == N.arange(0, new_duration, beat_length)[-1] and end:
                return "end"
            else:
                return ""
        def out_penalty(t):
            if t == 0 and start:
                return N.inf
            elif t == N.arange(0, new_duration, beat_length)[-1] and end:
                return N.inf
            else:
                return 0

        comp, info = retarget(song, new_duration, music_labels, out_labels, out_penalty)

        # and the beatless ending to the composition
        last_seg = sorted(comp.segments, key=lambda k: k.comp_location + k.duration)[-1]

        seg = Segment(song, comp.duration / float(song.samplerate),
            (last_seg.start + last_seg.duration) / float(song.samplerate),
            (song.duration - last_seg.start - last_seg.duration) / float(song.samplerate))
        comp.add_segment(seg)

    return comp


def retarget_with_change_points(song, cp_times, duration):
    """Create a composition of a song of a given duration that reaches
    music change points at specified times. This is still under
    construction- it<div></div> might not work as well with more than
    2 ``cp_times`` at the moment.

    Here's an example of retargeting music to be 40 seconds long and
    hit a change point at the 10 and 30 second marks::

        song = Song("instrumental_music.wav")
        composition = retarget.retarget_with_change_points(song, [10, 30], 40)
        composition.export(filename="retargeted_instrumental_music.")

    :param song: Song to retarget
    :type song: :py:class:`radiotool.composer.Song`
    :param cp_times: Times to reach change points (in seconds)
    :type cp_times: list of floats
    :param duration: Target length of retargeted music (in seconds)
    :type duration: float
    :returns: Composition of retargeted song
    :rtype: :py:class:`radiotool.composer.Composition`
    """
    analysis = song.analysis
    beat_length = analysis["avg_beat_duration"]

    # find change points
    cps = N.array(novelty(song, nchangepoints=4))
    cp_times = N.array(cp_times)

    # mark change points in original music
    def music_labels(t):
        if N.min(N.abs(cps - t)) < .5 * beat_length:
            return "cp"
        else:
            return "noncp"

    # mark where we want change points in the output music
    # (a few beats of slack to improve the quality of the end result)
    def out_labels(t):
        if N.min(N.abs(cp_times - t)) < 1.5 * beat_length:
            return "cp"
        return "noncp"

    # lower penalty around the target locations for change points
    # because we don't actually want each of them to be change points-
    # we just want one of the 4 beats covered to be a change point.
    def out_penalty(t):
        if N.min(N.abs(cp_times - t)) < 1.5 * beat_length:
            return .25
        return 1.0

    comp, info = retarget(song, duration, music_labels, out_labels, out_penalty)

    for k, v in info.iteritems():
        print k, v

    return comp


def retarget(song, duration, music_labels=None, out_labels=None, out_penalty=None):
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

    # return a radiotool Composition
    comp, cf_locations = _generate_audio(song, beats, path)

    info = {
        "cost": N.min(res) / len(path),
        "path": path,
        "target_labels": target,
        "result_labels": result_labels,
        "transitions": cf_locations
    }

    return comp, info


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
    trans_cost = N.copy(analysis["dense_dist"])

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

    # no self-jumps
    N.fill_diagonal(trans_cost, N.inf)

    min_jump = 4

    # no jumps within min-jump
    if min_jump and min_jump > 0:
        total_len = N.shape(trans_cost)[0]
        for idx in range(total_len):
            for diag_idx in range(-(min_jump - 1), min_jump):
                if 0 < idx + diag_idx < total_len and diag_idx != 1:
                    trans_cost[idx, idx + diag_idx] = N.inf


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
    comp = Composition(channels=song.channels)

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

    cf_locations = []

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
            cf_locations.append(current_loc)

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
        (last_seg.comp_location + last_seg.duration) / float(song.samplerate),
        music_volume)
    comp.add_dynamic(volume)

    # cf durs?
    # durs

    return comp, cf_locations
