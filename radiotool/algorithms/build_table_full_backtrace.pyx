#cython: infer_types=True
#cython: boundscheck=True
#cython: wraparound=False
#cython: profile=True
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
    int max_beats_with_padding
    int all_full


cdef void get_tc_column(double[:, :] tc, int column, double[:] tc_column, int backward, Params p):
    cdef int tc_index = 0
    cdef int beat_seg_i = 0
    cdef int i, j

    if column >= p.p0_full:
        tc_index = p.p0 + (column - p.p0_full)
    else:
        tc_index = column % p.n_beats

    if not backward:
        for i in range(p.max_beats_with_padding):
            for j in range(p.p0):
                tc_column[i * p.n_beats + j] = tc[j, tc_index]

        for i in range(p.p0_full, p.all_full):
            tc_column[i] = tc[p.p0 + i - p.p0_full, tc_index]
    else:
        for i in range(p.max_beats_with_padding):
            for j in range(p.p0):
                tc_column[i * p.n_beats + j] = tc[tc_index, j]

        for i in range(p.p0_full, p.all_full):
            tc_column[i] = tc[tc_index, p.p0 + i - p.p0_full]

    #--- CONSTRAINTS ---#
    # * don't go to pause before minimum length music segment
    if (column == p.p0_full) and (not backward):
        for i in range(p.n_beats * p.min_beats):
            tc_column[i] += p.pen_val
    elif (column < p.n_beats * p.min_beats) and backward:
        tc_column[p.p0_full] += p.pen_val

    # * don't go to pause after maximum length music segment
    if (column == p.p0_full) and (not backward):
        for i in range(p.n_beats * p.max_beats, p.p0_full):
            tc_column[i] += p.pen_val
    elif (p.p0_full > column >= p.n_beats * p.max_beats) and backward:
        tc_column[p.p0_full] += p.pen_val

    # * after pause, don't go to non-first segment beat
    if (p.n_beats <= column < p.p0_full) and (not backward):
        for i in range(p.p0_full, p.all_full):
            tc_column[i] += p.pen_val
    elif (column >= p.p0_full) and backward:
        for i in range(p.n_beats, p.p0_full):
            tc_column[i] += p.pen_val

    # * don't move between beats the don't follow segment index
    if column < p.p0_full:
        for i in range(p.p0_full):
            tc_column[i] += p.pen_val

        beat_seg_i = column / p.n_beats

        if (beat_seg_i > 0) and (not backward):
            for i in range((beat_seg_i - 1) * p.n_beats, beat_seg_i * p.n_beats):
                tc_column[i] -= p.pen_val

        elif (beat_seg_i < p.max_beats_with_padding - 1) and backward:
            for i in range((beat_seg_i + 1) * p.n_beats, (beat_seg_i + 2) * p.n_beats):
                tc_column[i] -= p.pen_val

        # you're also allowed to move infinitely among the
        # last beat if max_beats is not set (== -1)
        if p.max_beats == -1 and (beat_seg_i == p.min_beats):
            for i in range(beat_seg_i * p.n_beats, (beat_seg_i + 1) * p.n_beats):
                tc_column[i] -= p.pen_val


cdef double get_pen_value(double[:, :] pen, int i, int l, int global_start_l, Params p):
    cdef int pen_index = 0
    if i >= p.p0_full:
        pen_index = p.n_beats + (i - p.p0_full)
    else:
        pen_index = i % p.n_beats
    cdef double new_pen = pen[pen_index, l]

    #--- CONSTRAINTS ---#
    # * don't start song in segment beat other than first
    if global_start_l == 0 and (p.n_beats <= i < p.p0_full):
        new_pen += p.pen_val

    return new_pen


cdef void get_pen_column(double[:, :] pen, int column, double[:] new_pen, int global_start_l, Params p):
    cdef int i, j

    for i in range(p.max_beats_with_padding):
        for j in range(p.p0):
            new_pen[i * p.n_beats + j] = pen[j, column]

    for i in range(p.p0_full, p.all_full):
        new_pen[i] = pen[p.p0 + i - p.p0_full, column]

    #--- CONSTRAINTS ---#
    # * don't start song in segment beat other than first
    if global_start_l == 0:
        for i in range(p.n_beats, p.p0_full):
            new_pen[i] += p.pen_val


cdef void backward_space_efficient_cost_with_duration_constraint(
    double[:, :] tc, double[:, :] pen, int start_beat, int end_beat, int global_start_l, Params p,
    double[:] cost, double[:] pen_val, double[:] vals_col, double[:] min_vals, int[:] global_path):
    
    cdef int l, idx, i, beat_seg_i, seg_start_beat, j, full_j, orig_beat_j, min_idx
    cdef double minval, tmpval

    cdef double *full_cost = <double *>malloc(p.all_full * pen.shape[1] * sizeof(double))
    cdef int *prev_node = <int *>malloc(p.all_full * pen.shape[1] * sizeof(int))

    print "full cost size", p.all_full, "x", pen.shape[1], "=", p.all_full * pen.shape[1]
    
    if not full_cost:
        raise MemoryError()
    if not prev_node:
        raise MemoryError()

    min_idx = 0
    cdef rowlen = pen.shape[1]

    print "1"

    # generate initial cost
    if end_beat != -1:
        cost[:] = 99999999  # N.inf
        cost[end_beat] = get_pen_value(pen, end_beat, pen.shape[1] - 1, global_start_l + pen.shape[1] - 1, p)
    else:
        get_pen_column(pen, pen.shape[1] - 1, cost, global_start_l + pen.shape[1] - 1, p)

    for i in range(cost.shape[0]):
        print i, i * rowlen + (pen.shape[1] - 1)
        full_cost[i * rowlen + (pen.shape[1] - 1)] = cost[i]
        # ??
        # prev_node[i, pen.shape[1] - 1] = 0

    print "2"

    # optimize
    # should this be  -2 or -1?
    for l in xrange(pen.shape[1] - 2, 0, -1):
        print "l", l

        get_pen_column(pen, l, pen_val, global_start_l + l, p)

        # categories of beats we could be at before this one
        print "a"
        # beat segment before min_beat
        for idx in range(p.n_beats * (p.min_beats - 1)):
            beat_seg_i = idx / p.n_beats
            orig_beat_i = idx % p.n_beats

            # could only be going to beat_seg_i + 1
            seg_start_beat = (beat_seg_i + 1) * p.n_beats
            minval = -1
            for j in range(p.n_beats):
                tmpval = tc[orig_beat_i, j] + pen_val[idx] + full_cost[(seg_start_beat + j) * rowlen + (l + 1)]
                if minval == -1 or tmpval < minval:
                    minval = tmpval
                    min_idx = seg_start_beat + j

            min_vals[idx] = minval

            full_cost[idx * rowlen + l] = minval
            prev_node[idx * rowlen + l] = min_idx
        print "b"
        # beat segment between min beat and max beat
        for idx in range(p.n_beats * (p.min_beats - 1), p.n_beats * (p.max_beats - 1)):
            beat_seg_i = idx / p.n_beats
            orig_beat_i = idx % p.n_beats

            # could be going to beat_seg_i + 1
            seg_start_beat = (beat_seg_i + 1) * p.n_beats
            minval = -1
            for j in range(p.n_beats):
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
        print "c"
        # max beat segment
        for idx in range(p.n_beats * (p.max_beats - 1), p.n_beats * p.max_beats):
            orig_beat_i = idx % p.n_beats

            # must be going to first pause beat
            min_vals[idx] = tc[orig_beat_i, p.p0] + pen_val[idx] + full_cost[p.p0_full * rowlen + (l + 1)]

            full_cost[idx * rowlen + l] = tc[orig_beat_i, p.p0] + pen_val[idx] + full_cost[p.p0_full * rowlen + (l + 1)]
            prev_node[idx * rowlen + l] = p.p0_full
        print "d"
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
        print "e"
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

    # in the first column
    print "prev node 50, 50"
    print prev_node[50 * rowlen + 50]


    free( full_cost )
    free( prev_node )


cdef void divide_and_conquer_cost_and_path(
    double[:, :] tc, double[:, :] pen, int start_beat, int end_beat, int offset,
    int[:] global_path, Params p,
    double[:] f, double[:] g, double[:] mv1, double[:] mv2,
    double[:] mv3, double[:] mv4, double[:] mv5, double[:] mv6):

    backward_space_efficient_cost_with_duration_constraint(
        tc, pen, -1, end_beat, 0, p, g, mv4, mv5, mv6, global_path)

    return


cpdef int[:] build_table(double[:, :] trans_cost, double[:, :] penalty,
    int min_beats=-1, int max_beats=-1, int first_pause=-1):
    
    cdef int max_beats_with_padding, i

    if max_beats != -1 and min_beats != -1:
        # max_beats_with_padding = min_beats + max_beats
        max_beats_with_padding = max_beats
    elif max_beats != -1:
        # 4? One measures of padding? Just a thought
        max_beats_with_padding = max_beats
    elif min_beats != -1:
        max_beats = -1
        max_beats_with_padding = min_beats
    else:
        max_beats_with_padding = 1
        max_beats = 1
        min_beats = 0

    cdef Params p
    p.pen_val = 99999999.0
    p.p0 = first_pause
    p.n_beats = p.p0
    p.n_pauses = trans_cost.shape[0] - p.p0
    p.min_beats = min_beats
    p.max_beats = max_beats
    p.max_beats_with_padding = max_beats_with_padding
    p.p0_full = p.n_beats * p.max_beats_with_padding
    p.all_full = p.p0_full + p.n_pauses

    # double arrays for use throughout the computation
    cdef array dtemplate = array('d')
    cdef array array1, array2, array3, array4, array5, array6, array7, array8
    cdef double[:] mv1, mv2, mv3, mv4, mv5, mv6, f, g
    array1 = clone(dtemplate, p.all_full, False)
    array2 = clone(dtemplate, p.all_full, False)
    array3 = clone(dtemplate, p.all_full, False)
    array4 = clone(dtemplate, p.all_full, False)
    array5 = clone(dtemplate, p.all_full, False)
    array6 = clone(dtemplate, p.all_full, False)
    array7 = clone(dtemplate, p.all_full, False)
    array8 = clone(dtemplate, p.all_full, False)
    f = array1
    g = array2
    mv1 = array3
    mv2 = array4
    mv3 = array5
    mv4 = array6
    mv5 = array7
    mv6 = array8

    cdef array ar, template = array('i')
    ar = clone(template, penalty.shape[1], False)
    cdef int[:] global_path = ar

    divide_and_conquer_cost_and_path(trans_cost, penalty, -1, -1, 0, global_path, p,
        f, g, mv1, mv2, mv3, mv4, mv5, mv6)

    return global_path
