#cython: infer_types=True
#cython: boundscheck=True
#cython: wraparound=False
#cython: profile=True
#cython: cdivision=True
import cython
from cpython.array cimport array, clone

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


cdef void space_efficient_cost_with_duration_constraint(
    double[:, :] tc, double[:, :] pen, int start_beat, int end_beat, int global_start_l, Params p,
    double[:] cost, double[:] pen_val, double[:] vals_col, double[:] min_vals):

    cdef int l, idx, i, beat_seg_i, seg_start_beat, j, full_j, orig_beat_j
    cdef double minval, tmpval

    # generate initial cost
    if start_beat != -1:
        cost[:] = 99999999  # N.inf
        cost[start_beat] = get_pen_value(pen, start_beat, 0, global_start_l, p)
    else:
        get_pen_column(pen, 0, cost, global_start_l, p)

    # optimize
    for l in range(1, pen.shape[1]):
        if l == pen.shape[1] - 1 and end_beat != -1:
            # handle end beat set
            end_pen = get_pen_value(pen, end_beat, l, global_start_l + l, p)
            get_tc_column(tc, end_beat, vals_col, 0, p)

            min_vals[:] = 99999999  # N.inf
            minval = -1
            for i in range(vals_col.shape[0]):
                if minval == -1 or vals_col[i] + cost[i] + end_pen < minval:
                    minval = vals_col[i] + cost[i] + end_pen

            min_vals[end_beat] = minval

        else:
            get_pen_column(pen, l, pen_val, global_start_l + l, p)

            # Based on the nature of our problem
            # we have a HARD CONSTRAINT that the music
            # must move forward in the beat segment index.

            # This means that we don't need to check any of the
            # transitions except those that move to the next
            # beat segment index, or that go to a pause.
            # Or if we're in a pause, those that go to 
            # other pauses or go to the first beat segment index.

            # categories of beats we could be going to

            # first beat segment
            for idx in range(p.n_beats):
                # could only get here from the last pause beat
                min_vals[idx] = tc[p.n_beats + p.n_pauses - 1, idx] + pen_val[idx] + cost[p.all_full - 1]

            # all other music beat segments
            for idx in range(p.n_beats, p.p0_full):
                beat_seg_i = idx / p.n_beats
                orig_beat_i = idx % p.n_beats

                # must have gotten here from beat_seg_i - 1
                # and minimum value will be min cost from
                # another music beat
                seg_start_beat = (beat_seg_i - 1) * p.n_beats
                minval = -1
                for j in range(p.n_beats):
                    tmpval = tc[j, orig_beat_i] + pen_val[idx] + cost[seg_start_beat + j]
                    if minval == -1 or tmpval < minval:
                        minval = tmpval

                min_vals[idx] = minval

            # first pause beat:
            # must have gotten here from 
            # min beat <= beat seg <= max beat
            minval = -1
            for full_j in range(p.n_beats * (p.min_beats - 1), p.n_beats * p.max_beats):
                orig_beat_j = full_j % p.n_beats
                tmpval = tc[orig_beat_j, p.p0] + pen_val[p.p0_full] + cost[full_j]
                if minval == -1 or tmpval < minval:
                    minval = tmpval
            min_vals[p.p0_full] = minval

            # other pause beat
            for idx in range(p.p0_full + 1, p.all_full):
                orig_beat_i = p.p0 + (idx - p.p0_full)

                # must have gotten here from another pause beat
                minval = -1
                for j in range(p.n_pauses):
                    tmpval = tc[p.p0 + j, orig_beat_i] + pen_val[idx] + cost[p.p0_full + j]
                    if minval == -1 or tmpval < minval:
                        minval = tmpval
                min_vals[idx] = minval

        cost[:] = min_vals


cdef void backward_space_efficient_cost_with_duration_constraint(
    double[:, :] tc, double[:, :] pen, int start_beat, int end_beat, int global_start_l, Params p,
    double[:] cost, double[:] pen_val, double[:] vals_col, double[:] min_vals):
    
    cdef int l, idx, i, beat_seg_i, seg_start_beat, j, full_j, orig_beat_j
    cdef double minval, tmpval
    
    # generate initial cost
    if end_beat != -1:
        cost[:] = 99999999  # N.inf
        cost[end_beat] = get_pen_value(pen, end_beat, pen.shape[1] - 1, global_start_l + pen.shape[1] - 1, p)
    else:
        get_pen_column(pen, pen.shape[1] - 1, cost, global_start_l + pen.shape[1] - 1, p)

    # optimize
    for l in xrange(pen.shape[1] - 1, 0, -1):
        if l == 0 and start_beat != -1:
            # handle start beat set
            start_pen = get_pen_value(pen, start_beat, l, global_start_l + l, p)
            get_tc_column(tc, start_beat, vals_col, 1, p)

            min_vals[:] = 99999999  # N.inf
            minval = -1
            for i in range(vals_col.shape[0]):
                if minval == -1 or vals_col[i] + cost[i] + start_pen < minval:
                    minval = vals_col[i] + cost[i] + start_pen

            min_vals[start_beat] = minval

        else:
            get_pen_column(pen, l, pen_val, global_start_l + l, p)

            # categories of beats we could be at before this one

            # beat segment before min_beat
            for idx in range(p.n_beats * (p.min_beats - 1)):
                beat_seg_i = idx / p.n_beats
                orig_beat_i = idx % p.n_beats

                # could only be going to beat_seg_i + 1
                seg_start_beat = (beat_seg_i + 1) * p.n_beats
                minval = -1
                for j in range(p.n_beats):
                    tmpval = tc[orig_beat_i, j] + pen_val[idx] + cost[seg_start_beat + j]
                    if minval == -1 or tmpval < minval:
                        minval = tmpval

                min_vals[idx] = minval

            # beat segment between min beat and max beat
            for idx in range(p.n_beats * (p.min_beats - 1), p.n_beats * (p.max_beats - 1)):
                beat_seg_i = idx / p.n_beats
                orig_beat_i = idx % p.n_beats

                # could be going to beat_seg_i + 1
                seg_start_beat = (beat_seg_i + 1) * p.n_beats
                minval = -1
                for j in range(p.n_beats):
                    tmpval = tc[orig_beat_i, j] + pen_val[idx] + cost[seg_start_beat + j]
                    if minval == -1 or tmpval < minval:
                        minval = tmpval
                # or could be going to first pause beat
                tmpval = tc[orig_beat_i, p.p0] + pen_val[idx] + cost[p.p0_full]
                if minval == -1 or tmpval < minval:
                    minval = tmpval

                min_vals[idx] = minval

            # max beat segment
            for idx in range(p.n_beats * (p.max_beats - 1), p.n_beats * p.max_beats):
                orig_beat_i = idx % p.n_beats

                # must be going to first pause beat
                min_vals[idx] = tc[orig_beat_i, p.p0] + pen_val[idx] + cost[p.p0_full]

            # pause beats except the last one
            for idx in range(p.p0_full, p.all_full - 1):
                orig_beat_i = p.p0 + (idx - p.p0_full)

                # could only be going to another pause beat
                minval = -1
                for j in range(p.n_pauses):
                    tmpval = tc[orig_beat_i, p.p0 + j] + pen_val[idx] + cost[p.p0_full + j]
                    if minval == -1 or tmpval < minval:
                        minval = tmpval
                min_vals[idx] = minval

            # last pause beat
            minval = -1
            for j in range(p.n_beats):
                tmpval = tc[p.p0 + p.n_pauses - 1, j] + pen_val[p.all_full - 1] + cost[j]
                if minval == -1 or tmpval < minval:
                    minval = tmpval
            min_vals[p.all_full - 1] = minval

        cost[:] = min_vals


cdef void divide_and_conquer_cost_and_path(
    double[:, :] tc, double[:, :] pen, int start_beat, int end_beat, int offset,
    int[:] global_path, Params p,
    double[:] f, double[:] g, double[:] mv1, double[:] mv2,
    double[:] mv3, double[:] mv4, double[:] mv5, double[:] mv6):

    cdef int l = pen.shape[1]  # out beats
    cdef double[:] new_pen, tc_column
    cdef int i, opt_i, l_over_2
    cdef double minval = -1.0

    if l == 0:
        pass
    elif l == 1:
        pass
    elif l == 2 and start_beat != -1 and end_beat != -1:
        pass
    elif l == 2 and start_beat != -1:
        new_pen = mv1
        get_pen_column(pen, 1, new_pen, offset, p)

        tc_column = mv2
        get_tc_column(tc, start_beat, tc_column, 1, p)

        global_path[offset] = start_beat

        minval = -1.0
        opt_i = 0
        for i in range(tc_column.shape[0]):
            if minval == -1.0 or tc_column[i] + new_pen[i] < minval:
                minval = tc_column[i] + new_pen[i]
                opt_i = i

        global_path[offset + 1] = opt_i

        # global_path_cost[offset + 1] = N.min(tc_column + new_pen)
    elif l == 2 and end_beat != -1:
        new_pen = mv1
        get_pen_column(pen, 0, new_pen, offset, p)

        tc_column = mv2
        get_tc_column(tc, end_beat, tc_column, 0, p)

        minval = -1.0
        opt_i = 0
        for i in range(tc_column.shape[0]):
            if minval == -1.0 or tc_column[i] + new_pen[i] < minval:
                minval = tc_column[i] + new_pen[i]
                opt_i = i

        global_path[offset] = opt_i
        global_path[offset + 1] = end_beat

        # global_path_cost[offset] = N.min(tc_column + new_pen)
    elif l == 2:
        pass
        # opt_path = cost_and_path(tc, pen, start_beat, end_beat)
        # global_path[offset:offset + pen.shape[1]] = opt_path

    else:
        l_over_2 = l / 2

        # not sure why we need 8 of these arrays instead of 4
        space_efficient_cost_with_duration_constraint(
            tc, pen[:, :l_over_2 + 1], start_beat, -1, offset, p, f, mv1, mv2, mv3)

        backward_space_efficient_cost_with_duration_constraint(
            tc, pen[:, l_over_2:], -1, end_beat, offset + l_over_2, p, g, mv4, mv5, mv6)

        minval = -1.0
        opt_i = 0
        for i in range(f.shape[0]):
            if minval == -1.0 or f[i] + g[i] < minval:
                minval = f[i] + g[i]
                opt_i = i

        global_path[l_over_2 + offset] = opt_i
        # global_path_cost[l_over_2 + offset] = N.min(f + g)

        # first half
        divide_and_conquer_cost_and_path(
            tc, pen[:, :l_over_2 + 1], start_beat, opt_i, offset, global_path, p,
            f, g, mv1, mv2, mv3, mv4, mv5, mv6)

        # second half
        divide_and_conquer_cost_and_path(
            tc, pen[:, l_over_2:], opt_i, end_beat, l_over_2 + offset, global_path, p,
            f, g, mv1, mv2, mv3, mv4, mv5, mv6)

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
