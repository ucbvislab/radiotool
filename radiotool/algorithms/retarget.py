import copy
from collections import namedtuple

import numpy as N

from ..composer import Composition, Segment, Volume, Label, RawVolume
from novelty import novelty
from . import build_table
from . import build_table_mem_efficient
from . import par_build_table
import constraints

Spring = namedtuple('Spring', ['time', 'duration'])

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

        labels = []
        for transition in info["transitions"]:
            labels.append(Label("crossfade", transition))
        comp.add_labels(labels)

        # and the beatless ending to the composition
        last_seg = sorted(comp.segments, key=lambda k: k.comp_location + k.duration)[-1]

        seg = Segment(song, comp.duration_in_seconds,
            last_seg.start_in_seconds + last_seg.duration_in_seconds,
            song.duration_in_seconds - last_seg.start_in_seconds - last_seg.duration_in_seconds)
        comp.add_segment(seg)


    for transition in info["transitions"]:
        comp.add_label(Label("crossfade", transition))
    return comp


def retarget_with_change_points(song, cp_times, duration):
    """Create a composition of a song of a given duration that reaches
    music change points at specified times. This is still under
    construction. It might not work as well with more than
    2 ``cp_times`` at the moment.

    Here's an example of retargeting music to be 40 seconds long and
    hit a change point at the 10 and 30 second marks::

        song = Song("instrumental_music.wav")
        composition, change_points = retarget.retarget_with_change_points(song, [10, 30], 40)
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

    beat_length = analysis["avg_beat_duration"]
    beats = N.array(analysis["beats"])

    # find change points
    cps = N.array(novelty(song, nchangepoints=4))
    cp_times = N.array(cp_times)

    # mark change points in original music
    def music_labels(t):
        # find beat closest to t
        closest_beat_idx = N.argmin(N.abs(beats - t))
        closest_beat = beats[closest_beat_idx]
        closest_cp = cps[N.argmin(N.abs(cps - closest_beat))]
        
        if N.argmin(N.abs(beats - closest_cp)) == closest_beat_idx:
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

    final_cp_locations = [beat_length * i
                          for i, label in enumerate(info['result_labels'])
                          if label == 'cp']

    return comp, final_cp_locations

@profile
def retarget(song, duration, music_labels=None, out_labels=None, out_penalty=None,
             volume=None, volume_breakpoints=None, springs=None):
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
        pen = N.array([out_penalty(i) for i in N.arange(0, duration, beat_length)])
    else:
        pen = N.array([1 for i in N.arange(0, duration, beat_length)])
    
    pipeline = constraints.ConstraintPipeline(constraints=(
        constraints.PauseConstraint(6, 25),
        constraints.PauseEntryLabelChangeConstraint(target, .005),
        constraints.PauseExitLabelChangeConstraint(target, .005),
        constraints.TimbrePitchConstraint(context=2),
        constraints.EnergyConstraint(),
        # constraints.RhythmConstraint(3, 5.0),  # get time signature?
        constraints.MinimumJumpConstraint(8),
        constraints.LabelConstraint(start, target, pen),
        constraints.NoveltyConstraint(start, target, pen),
    ))

    trans_cost, penalty, beat_names = pipeline.apply(song, len(target))
    # trans_cost2, penalty2, beat_names2 = pipeline2.apply(song, len(target))

    print "Building cost table"

    # fortran method
    # cost2, prev_node2 = build_table(trans_cost2, penalty2)
    # res = cost2[:, -1]
    # best_idx = N.argmin(res)
    # if N.isfinite(res[best_idx]):
    #     path2, path_cost2 = _reconstruct_path(
    #         prev_node2, cost2, beat_names2, best_idx, N.shape(cost2)[1] - 1)
    #     opt_path2 = [beat_names2.index(x) for x in path2]
    # else:
    #     # throw an exception here?
    #     return None

    # path_cost2 = []
    # for i, node in enumerate(path2):
    #     if i == 0:
    #         path_cost2.append(0)
    #     else:
    #         path_cost2.append(trans_cost2[path2[i - 1], node] + penalty2[node, i])
    # path_cost2 = N.array(path_cost2)



    # compute the dynamic programming table (prev python method)
    # cost, prev_node = _build_table(analysis, duration, start, target, pen)

    # forward/backward memory efficient method
    first_pause = 0
    for i, bn in enumerate(beat_names):
        if str(bn).startswith('p'):
            first_pause = i
            break

    max_beats = 64
    min_beats = 4

    # max_beats = min(max_beats, penalty.shape[1])

    # path2_i, path2_cost = _build_table_forward_backward(trans_cost2, penalty2,
    #     first_pause=first_pause, max_beats=max_beats, min_beats=min_beats)

    tc2 = N.nan_to_num(trans_cost)
    pen2 = N.nan_to_num(penalty)


    if max_beats is not None and min_beats is not None:
        print "Running optimization (parallel, memory efficient)"

        # path2_i = build_table_mem_efficient(tc2, pen2,
        #     first_pause=first_pause, max_beats=max_beats, min_beats=min_beats)

        path_i = par_build_table(tc2, pen2,
            first_pause=first_pause, max_beats=max_beats, min_beats=min_beats)

        path = []
        first_pause_full = (max_beats + min_beats) * first_pause
        n_beats = first_pause
        for i in path_i:
            if i >= first_pause_full:
                path.append('p' + str(i - first_pause_full))
            else:
                path.append(float(beat_names[i % n_beats]))

        # need to compute path cost in the forward/backward method
        # because of changing duration constraints
        path_cost = N.zeros(path_i.shape)

        import pdb; pdb.set_trace()

    else:
        print "Running optimization (fast, full table)"
        # fortran method
        cost, prev_node = build_table(trans_cost, penalty)
        res = cost[:, -1]
        best_idx = N.argmin(res)
        if N.isfinite(res[best_idx]):
            path, path_cost = _reconstruct_path(
                prev_node, cost, beat_names, best_idx, N.shape(cost)[1] - 1)
            path_i = [beat_names.index(x) for x in path]
        else:
            # throw an exception here?
            return None

        path_cost = []
        for i, node in enumerate(path_i):
            if i == 0:
                path_cost.append(0)
            else:
                path_cost.append(trans_cost[path_i[i - 1], node] + penalty[node, i])
        path_cost = N.array(path_cost)

    # how did we do?
    result_labels = []
    for i in path:
        if str(i).startswith('p'):
            result_labels.append(None)
        else:
            result_labels.append(start[N.where(N.array(beats) == i)[0][0]])
    # result_labels = [start[N.where(N.array(beats) == i)[0][0]] for i in path]


    # return a radiotool Composition
    print "Generating audio"
    comp, cf_locations, result_full_labels, cost_labels, contracted = _generate_audio(
        song, beats, path, path_cost, start,
        volume=volume,
        volume_breakpoints=volume_breakpoints,
        springs=springs)

    info = {
        "contracted": contracted,
        "cost": N.sum(path_cost) / len(path),
        "path": path,
        "target_labels": target,
        "result_labels": result_labels,
        "result_full_labels": result_full_labels,
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

    return beat_path, path_cost


def _build_table_from_costs(trans_cost, penalty):
    # create cost matrix
    cost = N.zeros(penalty.shape)
    prev_node = N.zeros(penalty.shape)

    cost[:, 0] = penalty[:, 0]

    for l in xrange(1, penalty.shape[1]):
        tc = penalty[:, l] + trans_cost + cost[:, l - 1][:, N.newaxis]
        min_nodes = __fast_argmin_axis_0(tc)
        min_vals = N.amin(tc, axis=0)
        cost[:, l] = min_vals
        prev_node[:, l] = min_nodes

    return cost, prev_node


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
        tc = penalty[:, l] + trans_cost + cost[:, l - 1][:, N.newaxis]
        min_nodes = __fast_argmin_axis_0(tc)
        min_vals = N.amin(tc, axis=0)
        cost[:, l] = min_vals
        prev_node[:, l] = min_nodes

        # for n_i in xrange(len(beats)):
        #     total_cost = penalty[n_i, l] + trans_cost[:, n_i] + cost[:, l - 1]
        #     min_node = N.argmin(total_cost)
        #     cost[n_i, l] = total_cost[min_node]
        #     prev_node[n_i, l] = min_node

    # result:
    return cost, prev_node

def __fast_argmin_axis_0(a):
    # http://stackoverflow.com/questions/17840661/is-there-a-way-to-make-numpy-argmin-as-fast-as-min
    matches = N.nonzero((a == N.min(a, axis=0)).ravel())[0]
    rows, cols = N.unravel_index(matches, a.shape)
    argmin_array = N.empty(a.shape[1], dtype=N.intp)
    argmin_array[cols] = rows
    return argmin_array


def _generate_audio(song, beats, new_beats, new_beats_cost, music_labels,
                    volume=None, volume_breakpoints=None,
                    springs=None):
    print "Building volume"
    if volume is not None and volume_breakpoints is not None:
        raise Exception("volume and volume_breakpoints cannot both be defined")
    if volume is None and volume_breakpoints is None:
        volume = 1.0
    if volume_breakpoints is not None:
        volume_array = volume_breakpoints.to_array(song.samplerate)

    comp = Composition(channels=song.channels)

    audio_segments = []
    current_seg = [0, 0]
    if str(new_beats[0]).startswith('p'):
        current_seg = 'p'

    for i, b in enumerate(new_beats):
        if current_seg == 'p' and not str(b).startswith('p'):
            current_seg = [i, i]
        elif current_seg != 'p' and str(b).startswith('p'):
            audio_segments.append(current_seg)
            current_seg = 'p'
        elif current_seg != 'p':
            current_seg[1] = i
    if current_seg != 'p':
        audio_segments.append(current_seg)

    beats = N.array(beats)
    score_start = 0
    current_loc = 0.0
    last_segment_beat = 0

    comp.add_track(song)

    all_cf_locations = []

    print "Building audio"
    for aseg in audio_segments:
        segments = []
        starts = N.array(new_beats[aseg[0]:aseg[1] + 1])

        bis = [N.nonzero(beats == b)[0][0] for b in starts]
        dists = N.zeros(len(starts))
        durs = N.zeros(len(starts))

        for i, beat in enumerate(starts):
            if i < len(bis) - 1:
                if bis[i] + 1 != bis[i + 1]:
                    dists[i + 1] = 1
            if bis[i] + 1 >= len(beats):
                # use the average beat duration if we don't know
                # how long the beat is supposed to be
                print "USING BEAT DURATION IN SYNTHESIS - POTENTIALLY NOT GOOD"
                durs[i] = song.analysis["avg_beat_duration"]
            else:
                durs[i] = beats[bis[i] + 1] - beats[bis[i]]

        # add pause duration to current location
        current_loc += (aseg[0] - last_segment_beat) * song.analysis["avg_beat_duration"]
        last_segment_beat = aseg[1] + 1
        
        cf_durations = []
        seg_start = starts[0]
        seg_start_loc = current_loc

        cf_locations = []

        segment_starts = [0]
        try:
            segment_starts.extend(N.where(dists == 1)[0])
        except:
            pass

        # print "segment starts", segment_starts

        for i, s_i in enumerate(segment_starts):
            if i == len(segment_starts) - 1:
                # last segment?
                seg_duration = N.sum(durs[s_i:])
            else:
                next_s_i = segment_starts[i + 1]
                seg_duration = N.sum(durs[s_i:next_s_i])

                cf_durations.append(durs[next_s_i])
                cf_locations.append(current_loc + seg_duration)

            seg_music_location = starts[s_i]

            seg = Segment(song, current_loc, seg_music_location, seg_duration)
            # print "seg at", current_loc, "(music", seg_music_location, ") for", seg_duration
            # print "\twith beats", starts[s_i:next_s_i]
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

        for i, seg in enumerate(segments[:-1]):
            rawseg = comp.cross_fade(seg, segments[i + 1], cf_durations[i])

            # decrease volume along crossfades
            volume_frames = volume_array[
                rawseg.comp_location:rawseg.comp_location + rawseg.duration]
            raw_vol = RawVolume(rawseg, volume_frames)
            comp.add_dynamic(raw_vol)

        comp.fade_in(segments[0], 3.0)
        comp.fade_out(segments[-1], 3.0)

        prev_end = 0.0

        for seg in segments:
            volume_frames = volume_array[seg.comp_location:seg.comp_location + seg.duration]

            # this can happen on the final segment:
            if len(volume_frames) == 0:
                volume_frames = N.array([prev_end] * seg.duration)
            elif len(volume_frames) < seg.duration:
                delta = [volume_frames[-1]] * (seg.duration - len(volume_frames))
                volume_frames = N.r_[volume_frames, delta]
            raw_vol = RawVolume(seg, volume_frames)
            comp.add_dynamic(raw_vol)

            if len(volume_frames) != 0:
                prev_end = volume_frames[-1]

            # vol = Volume.from_segment(seg, volume)
            # comp.add_dynamic(vol)
            # print seg.comp_location_in_seconds, vol.comp_location_in_seconds, seg.duration == vol.duration

        all_cf_locations.extend(cf_locations)

    # result labels
    label_time = 0.0
    pause_len = song.analysis["avg_beat_duration"]
    result_full_labels = []
    prev_label = -1
    for beat in new_beats:
        if str(beat).startswith('p'):
            current_label = None
            if current_label != prev_label:
                result_full_labels.append(Label("pause", label_time))
            prev_label = None

            label_time += pause_len
        else:
            beat_i = N.where(N.array(beats) == beat)[0][0]
            next_i = beat_i + 1
            current_label = music_labels[beat_i]
            if current_label != prev_label:
                if current_label is None:
                    result_full_labels.append(Label("none", label_time))
                else:
                    result_full_labels.append(Label(current_label, label_time))
            prev_label = current_label

            if (next_i >= len(beats)):
                print "USING AVG BEAT DURATION - POTENTIALLY NOT GOOD"
                label_time += song.analysis["avg_beat_duration"]
            else:
                label_time += beats[next_i] - beat

    # result costs
    cost_time = 0.0
    pause_len = song.analysis["avg_beat_duration"]
    result_cost = []
    for i, b in enumerate(new_beats):
        result_cost.append(Label(new_beats_cost[i], cost_time))

        if str(b).startswith('p'):
            cost_time += pause_len
        else:
            beat_i = N.where(N.array(beats) == b)[0][0]
            next_i = beat_i + 1

            if (next_i >= len(beats)):
                cost_time += song.analysis["avg_beat_duration"]
            else:
                cost_time += beats[next_i] - b

    print "Contracting pause springs"
    contracted = []
    min_contraction = 0.5
    if springs is not None:
        offset = 0.0
        for spring in springs:
            contracted_time, contracted_dur = comp.contract(spring.time - offset, spring.duration,
                min_contraction=min_contraction)
            if contracted_dur > 0:
                new_cf = []
                for cf in all_cf_locations:
                    if cf > contracted_time:
                        new_cf.append(cf - contracted_dur)
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

                    if lab.time > contracted_time:
                        # TODO: fix this hack
                        if lab.name == "pause" and first_label:
                            pass
                        else:
                            lab.time -= contracted_dur
                        first_label = False

                # new_result_cost = []
                # for cost_lab in result_cost:
                #     if cost_lab.time < contracted_time:
                #         new_result_cost.append(cost_lab)
                #     elif contracted_time < cost_lab.time <=\
                #         contracted_time + contracted_dur:
                #         # remove cost labels in this range
                #         if cost_lab.name > 0:
                #             print "DELETING nonzero cost label", cost_lab.name, cost_lab.time
                #     else:
                #         cost_lab.time -= contracted_dur
                #         new_result_cost.append(cost_lab)


                new_result_cost = []
                first_label = True
                # TODO: also this hack. bleh.
                for cost_lab in result_cost:
                    if cost_lab.time < contracted_time:
                        new_result_cost.append(cost_lab)
                    elif cost_lab.time > contracted_time and\
                        cost_lab.time <= contracted_time + contracted_dur:
                        if first_label:
                            cost_lab.time = contracted_time
                            new_result_cost.append(cost_lab)
                        elif cost_lab.name > 0:
                            print "DELETING nonzero cost label:", cost_lab.name, cost_lab.time
                        first_label = False
                    elif cost_lab.time > contracted_time + contracted_dur:
                        cost_lab.time -= contracted_dur
                        new_result_cost.append(cost_lab)
                        first_label = False
                result_cost = new_result_cost

                contracted.append(Spring(contracted_time + offset, contracted_dur))
                offset += contracted_dur

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
    #     (last_seg.comp_location + last_seg.duration) / float(song.samplerate),
    #     volume)
    # comp.add_dynamic(vol)

    # cf durs?
    # durs

    return comp, all_cf_locations, result_full_labels, result_cost, contracted
