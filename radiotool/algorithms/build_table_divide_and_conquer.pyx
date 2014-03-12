#cython: infer_types=True
import cython
import numpy as N
cimport numpy as N

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


cdef get_tc_column(double[:, :] tc, int column, double[:] tc_column, int backward, Params p):
    cdef int tc_index = 0
    if column >= p.p0_full:
        tc_index = p.p0 + (column - p.p0_full)
    else:
        tc_index = column % p.n_beats

    if not backward:
        tc_column[:p.p0_full] = N.tile(tc[:p.p0, tc_index], p.max_beats_with_padding)
        tc_column[p.p0_full:] = tc[p.p0:, tc_index]
    else:
        tc_column[:p.p0_full] = N.tile(tc[tc_index, :p.p0], p.max_beats_with_padding)
        tc_column[p.p0_full:] = tc[tc_index, p.p0:]

    #--- CONSTRAINTS ---#
    # * don't go to pause before minimum length music segment
    if (column == p.p0_full) and (not backward):
        tc_column[:p.n_beats * p.min_beats] += p.pen_val
    elif (column < p.n_beats * p.min_beats) and backward:
        tc_column[p.p0_full] += p.pen_val

    # * don't go to pause after maximum length music segment
    if (column == p.p0_full) and (not backward):
        # print "changing (2)"
        tc_column[p.n_beats * p.max_beats:] += p.pen_val
    elif (column >= p.n_beats * p.max_beats) and backward:
        tc_column[p.p0_full] += p.pen_val

    # * after pause, don't go to non-first segment beat
    if (p.n_beats <= column < p.p0_full) and (not backward):
        # print "changing (3)"
        tc_column[p.p0_full:] += p.pen_val
    elif (column >= p.p0_full) and backward:
        tc_column[p.n_beats:p.p0_full] += p.pen_val


    # * don't move between beats the don't follow segment index
    cdef int beat_seg_i = 0
    if column < p.p0_full:
        tc_column[:p.p0_full] += p.pen_val
        beat_seg_i = int(column / float(p.n_beats))

        if (beat_seg_i > 0) and (not backward):
            tc_column[(beat_seg_i - 1) * p.n_beats:beat_seg_i * p.n_beats] -= p.pen_val

        elif (beat_seg_i < p.max_beats_with_padding - 1) and backward:
            tc_column[(beat_seg_i + 1) * p.n_beats:(beat_seg_i + 2) * p.n_beats] -= p.pen_val


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

cdef get_pen_column(double[:, :] pen, int column, double[:] new_pen, int global_start_l, Params p):
    new_pen[:p.p0_full] = N.tile(pen[:p.p0, column], p.max_beats_with_padding)
    new_pen[p.p0_full:] = pen[p.p0:, column] 

    #--- CONSTRAINTS ---#
    # * don't start song in segment beat other than first
    if global_start_l == 0:
        new_pen[p.n_beats:p.p0_full] += p.pen_val

cdef double[:] space_efficient_cost_with_duration_constraint(
    double[:, :] tc, double[:, :] pen, int start_beat, int end_beat, int global_start_l, Params p):

    cdef double[:] pen_val = N.empty(p.all_full)
    cdef double[:] cost = N.empty(p.all_full)
    cdef double[:] vals_col = N.empty(p.all_full)
    cdef double[:] min_vals = N.empty(p.all_full)
    cdef int l, idx

    # generate initial cost
    if start_beat != -1:
        cost[:] = N.inf
        cost[start_beat] = get_pen_value(pen, start_beat, 0, global_start_l, p)
    else:
        get_pen_column(pen, 0, cost, global_start_l, p)

    # optimize
    for l in range(1, pen.shape[1]):
        if l == pen.shape[1] - 1 and end_beat != -1:
            # handle end beat set
            end_pen = get_pen_value(pen, end_beat, l, global_start_l + l, p)
            get_tc_column(tc, end_beat, vals_col, 0, p)

            min_vals[:] = N.inf
            min_vals[end_beat] = N.min(N.add(N.add(vals_col, cost), end_pen))

        else:
            get_pen_column(pen, l, pen_val, global_start_l + l, p)
            for idx in range(p.all_full):
                get_tc_column(tc, idx, vals_col, 0, p)
                vals_col += N.add(cost, pen_val[idx])
                min_vals[idx] = N.min(vals_col)

        cost[:] = min_vals

    return cost

cdef double[:] backward_space_efficient_cost_with_duration_constraint(
    double[:, :] tc, double[:, :] pen, int start_beat, int end_beat, int global_start_l, Params p):
    
    cdef double[:] pen_val = N.empty(p.all_full)
    cdef double[:] cost = N.empty(p.all_full)
    cdef double[:] vals_col = N.empty(p.all_full)
    cdef double[:] min_vals = N.empty(p.all_full)
    cdef int l, idx
    
    # generate initial cost
    if end_beat != -1:
        cost[:] = N.inf
        cost[end_beat] = get_pen_value(pen, end_beat, pen.shape[1] - 1, global_start_l + pen.shape[1] - 1, p)
    else:
        get_pen_column(pen, pen.shape[1] - 1, cost, global_start_l + pen.shape[1] - 1, p)

    # optimize
    for l in xrange(1, pen.shape[1]):
        if l == 0 and start_beat != -1:
            # handle start beat set
            start_pen = get_pen_value(pen, end_beat, l, global_start_l + l, p)
            get_tc_column(tc, start_beat, vals_col, 1, p)

            min_vals[:] = N.inf
            min_vals[end_beat] = N.min(N.add(N.add(vals_col, cost), start_pen))

        else:
            get_pen_column(pen, l, pen_val, global_start_l + l, p)
            for idx in xrange(p.all_full):
                get_tc_column(tc, idx, vals_col, 1, p)
                vals_col += N.add(cost, pen_val[idx])
                min_vals[idx] = N.min(vals_col)

        cost[:] = min_vals

    return cost

cdef divide_and_conquer_cost_and_path(
    double[:, :] tc, double[:, :] pen, int start_beat, int end_beat, int offset,
    int[:] global_path, Params p):

    print "divide and conquer with", start_beat, end_beat, pen.shape[1]
    cdef int l = pen.shape[1]  # out beats
    cdef double[:] new_pen

    if l == 0: return
    elif l == 1: return
    elif l == 2 and start_beat != -1 and end_beat != -1: return
    elif l == 2 and start_beat != -1:
        new_pen = N.empty(p.all_full)
        get_pen_column(pen, 1, new_pen, offset, p)

        tc_column = N.empty(p.all_full)
        get_tc_column(tc, start_beat, tc_column, 1, p)

        global_path[offset] = start_beat
        global_path[offset + 1] = int(N.argmin(tc_column + new_pen))

        # global_path_cost[offset + 1] = N.min(tc_column + new_pen)
        return
    elif l == 2 and end_beat != -1:
        new_pen = N.empty(p.all_full)
        get_pen_column(pen, 0, new_pen, offset, p)

        tc_column = N.empty(p.all_full)
        get_tc_column(tc, end_beat, tc_column, 0, p)

        global_path[offset] = int(N.argmin(tc_column + new_pen))
        global_path[offset + 1] = end_beat

        # global_path_cost[offset] = N.min(tc_column + new_pen)
        return
    elif l == 2:
        print "actually running full optimize"
        # opt_path = cost_and_path(tc, pen, start_beat, end_beat)
        # global_path[offset:offset + pen.shape[1]] = opt_path
        return

    cdef int l_over_2 = int(float(l) / 2.0)

    cdef double[:] f =\
        space_efficient_cost_with_duration_constraint(tc, pen[:, :l_over_2 + 1], start_beat, -1, offset, p)
    cdef double[:] g =\
        backward_space_efficient_cost_with_duration_constraint(tc, pen[:, l_over_2:], -1, end_beat, offset + l_over_2, p)

    cdef int opt_i = int(N.argmin(N.add(f, g)))
    global_path[l_over_2 + offset] = opt_i
    # global_path_cost[l_over_2 + offset] = N.min(f + g)

    # first half
    divide_and_conquer_cost_and_path(
        tc, pen[:, :l_over_2 + 1], start_beat, opt_i, offset, global_path, p)

    # second half
    divide_and_conquer_cost_and_path(
        tc, pen[:, l_over_2:], opt_i, end_beat, l_over_2 + offset, global_path, p)

    return

cpdef int[:] _build_table_forward_backward(double[:, :] trans_cost, double[:, :] penalty,
    int min_beats=-1, int max_beats=-1, int first_pause=-1):

    cdef int max_beats_with_padding

    if max_beats != -1 and min_beats != -1:
        max_beats_with_padding = min_beats + max_beats
    elif max_beats != -1:
        # 8? Two measures of padding? Just a thought
        max_beats_with_padding = max_beats + 8
    else:
        max_beats_with_padding = 1
        max_beats = N.inf
        min_beats = 0

    cdef Params p
    p.pen_val = 1.0
    p.p0 = first_pause
    p.n_beats = trans_cost.shape[0] - p.p0
    p.n_pauses = trans_cost.shape[0] - p.p0
    p.min_beats = min_beats
    p.max_beats = max_beats
    p.max_beats_with_padding = max_beats_with_padding
    p.p0_full = p.n_beats * p.max_beats_with_padding
    p.all_full = p.p0_full + p.n_pauses

    # global_path_cost = N.zeros(penalty.shape[1])
    global_path = N.zeros(penalty.shape[1], dtype=N.int32)
    divide_and_conquer_cost_and_path(trans_cost, penalty, -1, -1, 0, global_path, p)

    return global_path