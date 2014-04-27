#cython: infer_types=True
#cython: boundscheck=True
#cython: wraparound=False
#cython: cdivision=True
import cython
from cpython.array cimport array, clone
from libc.stdlib cimport malloc, free

cdef struct Params:
    double pen_val
    int p0
    int p0_full
    int n_beats
    int n_pauses
    int min_beats
    int max_beats
    int all_full
    int no_max_beats
    int n_song_starts

cdef int MAX_BEATS = 0
cdef int NO_MAX_BEATS = 1

cdef double get_pen_value(double[:, :] pen, int i, int l, int global_start_l, Params p):
    cdef int pen_index = 0
    if i >= p.p0_full:
        pen_index = p.n_beats + (i - p.p0_full)
    else:
        pen_index = i % p.n_beats
    cdef double new_pen = pen[pen_index, l]

    if p.no_max_beats == MAX_BEATS:
        #--- CONSTRAINTS ---#
        # * don't start song in segment beat other than first
        if global_start_l == 0 and (p.n_beats <= i < p.p0_full):
            new_pen += p.pen_val

        # * don't end song in a segment beat other than beat past min_beats
        if global_start_l == pen.shape[1] - 1 and (i < p.n_beats * p.min_beats):
            new_pen += p.pen_val

    return new_pen


cdef void get_pen_column(double[:, :] pen, int column, double[:] new_pen, int global_start_l, Params p):
    cdef int i, j

    for i in range(p.max_beats):
        for j in range(p.p0):
            new_pen[i * p.n_beats + j] = pen[j, column]

    for i in range(p.p0_full, p.all_full):
        new_pen[i] = pen[p.p0 + i - p.p0_full, column]

    if p.no_max_beats == MAX_BEATS:
        #--- CONSTRAINTS ---#
        # * don't start song in segment beat other than first
        if global_start_l == 0:
            for i in range(p.n_beats, p.p0_full):
                new_pen[i] += p.pen_val

        # * don't end song in a segment beat other than beat past min_beats
        if global_start_l == pen.shape[1] - 1:
            for i in range(p.n_beats * p.min_beats):
                new_pen[i] += p.pen_val


cdef void backward_build_table(
    double[:, :] tc, double[:, :] pen, int start_beat, int end_beat, int global_start_l, Params p,
    int[:] song_starts, int[:] song_ends,
    double[:] cost, double[:] pen_val, double[:] vals_col, double[:] min_vals, int[:] global_path,
    double[:] global_path_cost):
    
    cdef int l, idx, i, beat_seg_i, seg_start_beat, j, full_j, orig_beat_j, min_idx, song_i
    cdef double minval, tmpval

    cdef double *full_cost = <double *>malloc(p.all_full * pen.shape[1] * sizeof(double))
    cdef int *prev_node = <int *>malloc(p.all_full * pen.shape[1] * sizeof(int))

    # print "full cost size", p.all_full, "x", pen.shape[1], "=", p.all_full * pen.shape[1]
    
    min_idx = 0
    cdef int rowlen = pen.shape[1]

    # generate initial cost
    if end_beat != -1:
        cost[:] = 99999999  # N.inf
        cost[end_beat] = get_pen_value(pen, end_beat, pen.shape[1] - 1, global_start_l + pen.shape[1] - 1, p)
    else:
        get_pen_column(pen, pen.shape[1] - 1, cost, global_start_l + pen.shape[1] - 1, p)

    for i in range(cost.shape[0]):
        # print i, i * rowlen + (pen.shape[1] - 1)
        full_cost[i * rowlen + (pen.shape[1] - 1)] = cost[i]
        # ??
        # prev_node[i, pen.shape[1] - 1] = 0

    # optimize
    # should this be  -2 or -1?
    for l in xrange(pen.shape[1] - 2, -1, -1):
        get_pen_column(pen, l, pen_val, global_start_l + l, p)

        # categories of beats we could be at before this one

        # beat segment before min_beat
        for idx in range(p.n_beats * (p.min_beats - 1)):
            beat_seg_i = idx / p.n_beats
            orig_beat_i = idx % p.n_beats

            song_i = 0
            for j in range(p.n_song_starts):
                if song_starts[j] <= orig_beat_i < song_ends[j]:
                    song_i = j
                    break

            # could only be going to beat_seg_i + 1
            seg_start_beat = (beat_seg_i + 1) * p.n_beats
            minval = -1
            for j in range(song_starts[song_i], song_ends[song_i]):
            # for j in range(p.n_beats):
                tmpval = tc[orig_beat_i, j] + pen_val[idx] + full_cost[(seg_start_beat + j) * rowlen + (l + 1)]
                if minval == -1 or tmpval < minval:
                    minval = tmpval
                    min_idx = seg_start_beat + j

            min_vals[idx] = minval

            full_cost[idx * rowlen + l] = minval
            prev_node[idx * rowlen + l] = min_idx

        if p.no_max_beats == MAX_BEATS:
            # beat segment between min beat and max beat
            for idx in range(p.n_beats * (p.min_beats - 1), p.n_beats * (p.max_beats - 1)):
                beat_seg_i = idx / p.n_beats
                orig_beat_i = idx % p.n_beats

                song_i = 0
                for j in range(p.n_song_starts):
                    if song_starts[j] <= orig_beat_i < song_ends[j]:
                        song_i = j
                        break

                # could be going to beat_seg_i + 1
                seg_start_beat = (beat_seg_i + 1) * p.n_beats

                minval = -1
                for j in range(song_starts[song_i], song_ends[song_i]):
                # for j in range(p.n_beats):
                    tmpval = tc[orig_beat_i, j] + pen_val[idx] + full_cost[(seg_start_beat + j) * rowlen + (l + 1)]
                    if minval == -1 or tmpval < minval:
                        minval = tmpval
                        min_idx = seg_start_beat + j
                # or could be going to first pause beat
                tmpval = tc[orig_beat_i, p.p0] + pen_val[idx] + full_cost[p.p0_full * rowlen + (l + 1)]
                if minval == -1 or tmpval < minval:
                    minval = tmpval
                    min_idx = p.p0_full

                min_vals[idx] = minval

                full_cost[idx * rowlen + l] = minval
                prev_node[idx * rowlen + l] = min_idx

            # max beat segment
            for idx in range(p.n_beats * (p.max_beats - 1), p.n_beats * p.max_beats):
                orig_beat_i = idx % p.n_beats

                # must be going to first pause beat
                min_vals[idx] = tc[orig_beat_i, p.p0] + pen_val[idx] + full_cost[p.p0_full * rowlen + (l + 1)]

                full_cost[idx * rowlen + l] = tc[orig_beat_i, p.p0] + pen_val[idx] + full_cost[p.p0_full * rowlen + (l + 1)]
                prev_node[idx * rowlen + l] = p.p0_full

        else:
            # no maximum beat

            # max beat segment
            for idx in range(p.n_beats * (p.max_beats - 1), p.n_beats * p.max_beats):
                beat_seg_i = idx / p.n_beats
                orig_beat_i = idx % p.n_beats

                song_i = 0
                for j in range(p.n_song_starts):
                    if song_starts[j] <= orig_beat_i < song_ends[j]:
                        song_i = j
                        break

                # could be going to same beat
                seg_start_beat = beat_seg_i

                minval = -1
                for j in range(song_starts[song_i], song_ends[song_i]):
                # for j in range(p.n_beats):
                    tmpval = tc[orig_beat_i, j] + pen_val[idx] + full_cost[(seg_start_beat + j) * rowlen + (l + 1)]
                    if minval == -1 or tmpval < minval:
                        minval = tmpval
                        min_idx = seg_start_beat + j

                if p.n_pauses > 0:
                    # or could be going to first pause beat
                    tmpval = tc[orig_beat_i, p.p0] + pen_val[idx] + full_cost[p.p0_full * rowlen + (l + 1)]
                    if minval == -1 or tmpval < minval:
                        minval = tmpval
                        min_idx = p.p0_full

                min_vals[idx] = minval

                full_cost[idx * rowlen + l] = minval
                prev_node[idx * rowlen + l] = min_idx

        # pause beats except the last one
        for idx in range(p.p0_full, p.all_full - 1):
            orig_beat_i = p.p0 + (idx - p.p0_full)

            # could only be going to another pause beat
            minval = -1
            for j in range(p.n_pauses):
                tmpval = tc[orig_beat_i, p.p0 + j] + pen_val[idx] + full_cost[(p.p0_full + j) * rowlen + (l + 1)]
                if minval == -1 or tmpval < minval:
                    minval = tmpval
                    min_idx = p.p0_full + j
            min_vals[idx] = minval

            full_cost[idx * rowlen + l] = minval
            prev_node[idx * rowlen + l] = min_idx

        if p.n_pauses > 0:
            # last pause beat
            minval = -1
            for j in range(p.n_beats):
                tmpval = tc[p.p0 + p.n_pauses - 1, j] + pen_val[p.all_full - 1] + full_cost[j * rowlen + (l + 1)]
                if minval == -1 or tmpval < minval:
                    minval = tmpval
                    min_idx = j
            min_vals[p.all_full - 1] = minval

        full_cost[(p.all_full - 1) * rowlen + l] = minval
        prev_node[(p.all_full - 1) * rowlen + l] = min_idx

        # cost[:] = min_vals

    # find the optimal path

    # find start node
    minval = -1
    for i in range(p.all_full):
        # print i, full_cost[i * rowlen + 0]
        if minval == -1 or full_cost[i * rowlen + 0] < minval:
            minval = full_cost[i * rowlen + 0]
            min_idx = i
    global_path[0] = min_idx

    cdef double total_cost_remaining = minval
    cdef int node = min_idx

    for l in range(pen.shape[1] - 1):
        node = prev_node[node * rowlen + l]
        global_path[l + 1] = node
        global_path_cost[l] = total_cost_remaining - full_cost[node * rowlen + (l + 1)]
        total_cost_remaining -= global_path_cost[l]

    # not sure what to grab for the initial cost
    global_path_cost[pen.shape[1] - 1] = total_cost_remaining

    free( full_cost )
    free( prev_node )


def build_table(double[:, :] trans_cost, double[:, :] penalty,
    int[:] song_starts, int[:] song_ends,
    int min_beats=-1, int max_beats=-1, int first_pause=-1):
    
    cdef int i

    cdef Params p

    if max_beats == -1:
        p.no_max_beats = NO_MAX_BEATS
        max_beats = min_beats + 1
    else:
        p.no_max_beats = MAX_BEATS

    p.pen_val = 99999999.0
    p.p0 = first_pause
    p.n_beats = p.p0
    p.n_pauses = trans_cost.shape[0] - p.p0
    p.min_beats = min_beats
    p.max_beats = max_beats
    p.p0_full = p.n_beats * p.max_beats
    p.all_full = p.p0_full + p.n_pauses

    p.n_song_starts = len(song_starts)

    # double arrays for use throughout the computation
    cdef array dtemplate = array('d')
    cdef array array1, array2, array3, array4, array5
    cdef double[:] mv1, mv2, mv3, mv4, global_path_cost
    array1 = clone(dtemplate, p.all_full, False)
    array2 = clone(dtemplate, p.all_full, False)
    array3 = clone(dtemplate, p.all_full, False)
    array4 = clone(dtemplate, p.all_full, False)
    mv1 = array1
    mv2 = array2
    mv3 = array3
    mv4 = array4

    array5 = clone(dtemplate, penalty.shape[1], False)
    global_path_cost = array5

    cdef array ar, template = array('i')
    ar = clone(template, penalty.shape[1], False)
    cdef int[:] global_path = ar

    backward_build_table(
        trans_cost, penalty, -1, -1, 0, p, song_starts, song_ends, mv1, mv2, mv3, mv4, global_path, global_path_cost)

    return [x for x in global_path], [x for x in global_path_cost]
