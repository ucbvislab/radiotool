# reference implementation for build_table_mem_efficient.pyx

def _build_table_forward_backward(trans_cost, penalty,
                                  min_beats=None, max_beats=None, first_pause=None):

    if max_beats is not None and min_beats is not None:
        max_beats_with_padding = min_beats + max_beats
    elif max_beats is not None:
        # 8? Two measures of padding? Just a thought
        max_beats_with_padding = max_beats + 8
    else:
        max_beats_with_padding = 1
        max_beats = N.inf
        min_beats = 0

    p0 = first_pause
    n_beats = first_pause
    n_pauses = trans_cost.shape[0] - p0
    p0_full = n_beats * max_beats_with_padding
    all_full = p0_full + n_pauses
    pen_val = 1.0

    def get_tc_column(tc, column, tc_column, backward=False):
        tc_index = 0
        if column >= p0_full:
            tc_index = p0 + (column - p0_full)
        else:
            tc_index = column % n_beats

        if not backward:
            tc_column[:p0_full] = N.tile(tc[:p0, tc_index], max_beats_with_padding)
            tc_column[p0_full:] = tc[p0:, tc_index]
        else:
            tc_column[:p0_full] = N.tile(tc[tc_index, :p0], max_beats_with_padding)
            tc_column[p0_full:] = tc[tc_index, p0:]

        #--- CONSTRAINTS ---#
        # * don't go to pause before minimum length music segment
        if (column == p0_full) and (not backward):
            tc_column[:n_beats * min_beats] += pen_val
        elif (column < n_beats * min_beats) and backward:
            tc_column[p0_full] += pen_val

        # * don't go to pause after maximum length music segment
        if (column == p0_full) and (not backward):
            # print "changing (2)"
            tc_column[n_beats * max_beats:] += pen_val
        elif (column >= n_beats * max_beats) and backward:
            tc_column[p0_full] += pen_val

        # * after pause, don't go to non-first segment beat
        if (n_beats <= column < p0_full) and (not backward):
            # print "changing (3)"
            tc_column[p0_full:] += pen_val
        elif (column >= p0_full) and backward:
            tc_column[n_beats:p0_full] += pen_val

        # * don't move between beats the don't follow segment index
        if column < p0_full:
            tc_column[:p0_full] += pen_val
            beat_seg_i = int(column / float(n_beats))

            if (beat_seg_i > 0) and (not backward):
                tc_column[(beat_seg_i - 1) * n_beats:beat_seg_i * n_beats] -= pen_val

            elif (beat_seg_i < max_beats_with_padding - 1) and backward:
                tc_column[(beat_seg_i + 1) * n_beats:(beat_seg_i + 2) * n_beats] -= pen_val


    def get_pen_value(pen, i, l, global_start_l):
        pen_index = 0
        if i >= p0_full:
            pen_index = n_beats + (i - p0_full)
        else:
            pen_index = i % n_beats
        new_pen = pen[pen_index, l]

        #--- CONSTRAINTS ---#
        # * don't start song in segment beat other than first
        if global_start_l == 0 and (n_beats <= i < p0_full):
            new_pen += pen_val

        return new_pen


    def get_pen_column(pen, column, new_pen, global_start_l):
        new_pen[:p0_full] = N.tile(pen[:p0, column], max_beats_with_padding)
        new_pen[p0_full:] = pen[p0:, column] 

        #--- CONSTRAINTS ---#
        # * don't start song in segment beat other than first
        if global_start_l == 0:
            new_pen[n_beats:p0_full] += pen_val


    def space_efficient_cost_with_duration_constraint(tc, pen, start_beat, end_beat, global_start_l):
        pen_val = N.empty(all_full)
        cost = N.empty(all_full)

        # generate initial cost
        if start_beat is not None:
            cost[:] = N.inf
            cost[start_beat] = get_pen_value(pen, start_beat, 0, global_start_l)
        else:
            get_pen_column(pen, 0, cost, global_start_l)

        # optimize
        vals_col = N.empty(all_full)
        min_vals = N.empty(all_full)
        for l in xrange(1, pen.shape[1]):
            if l == pen.shape[1] - 1 and end_beat is not None:
                # handle end beat set
                end_pen = get_pen_value(pen, end_beat, l, global_start_l + l)
                get_tc_column(tc, end_beat, vals_col)

                min_vals[:] = N.inf
                min_vals[end_beat] = N.min(vals_col + cost + end_pen)

            else:
                get_pen_column(pen, l, pen_val, global_start_l + l)
                for idx in xrange(all_full):
                    get_tc_column(tc, idx, vals_col)
                    vals_col += cost + pen_val[idx]
                    # vals_col += cost + get_pen_value(pen, idx, l, global_start_l + l)
                    min_vals[idx] = N.min(vals_col)

            cost[:] = min_vals

        return cost


    def backward_space_efficient_cost_with_duration_constraint(tc, pen, start_beat, end_beat, global_start_l):
        pen_val = N.empty(all_full)
        cost = N.empty(all_full)

        # generate initial cost
        if end_beat is not None:
            cost[:] = N.inf
            cost[end_beat] = get_pen_value(pen, end_beat, pen.shape[1] - 1, global_start_l + pen.shape[1] - 1)
        else:
            get_pen_column(pen, pen.shape[1] - 1, cost, global_start_l + pen.shape[1] - 1)

        # optimize
        vals_col = N.empty(all_full)
        min_vals = N.empty(all_full)
        for l in xrange(1, pen.shape[1]):
            if l == 0 and start_beat is not None:
                # handle start beat set
                start_pen = get_pen_value(pen, start_beat, l, global_start_l + l)
                get_tc_column(tc, start_beat, vals_col, backward=True)

                min_vals[:] = N.inf
                min_vals[start_beat] = N.min(vals_col + cost + start_pen)

            else:
                get_pen_column(pen, l, pen_val, global_start_l + l)
                for idx in xrange(all_full):
                    get_tc_column(tc, idx, vals_col, backward=True)
                    vals_col += cost + pen_val[idx]
                    # vals_col += cost + get_pen_value(pen, idx, l, global_start_l + l)
                    min_vals[idx] = N.min(vals_col)

            cost[:] = min_vals

        return cost


    def space_efficient_cost(tc, pen, start_beat, end_beat):
        cost = pen[:, 0]
        if start_beat is not None:
            cost = N.ones(pen.shape[0]) * N.inf
            cost[start_beat] = pen[start_beat, 0]
        for l in xrange(1, pen.shape[1]):
            p = pen[:, l]
            if l == pen.shape[1] - 1 and end_beat is not None:
                p = N.ones(pen.shape[0]) * N.inf
                p[end_beat] = pen[end_beat, -1]
            vals = p + tc + cost[:, N.newaxis]
            min_vals = N.amin(vals, axis=0)
            cost = min_vals

        return cost


    def backward_space_efficient_cost(tc, pen, start_beat, end_beat):
        cost = pen[:, -1]
        if end_beat is not None:
            cost = N.ones(pen.shape[0]) * N.inf
            cost[end_beat] = pen[end_beat, -1]
        for l in xrange(pen.shape[1] - 2, -1, -1):
            p = pen[:, l]
            if l == 0 and start_beat is not None:
                p = N.ones(pen.shape[0]) * N.inf
                p[start_beat] = pen[start_beat, 0]
            vals = p + tc.T + cost[:, N.newaxis]
            min_vals = N.amin(vals, axis=0)
            cost = min_vals

        return cost


    def cost_and_path(tc, pen, start_beat, end_beat):
        cost = N.zeros(pen.shape)
        prev_node = N.zeros(pen.shape)

        cost[:, 0] = pen[:, 0]
        if start_beat is not None:
            cost[:, 0] = N.ones(pen.shape[0]) * N.inf
            cost[start_beat, 0] = pen[start_beat, 0]

        for l in xrange(1, pen.shape[1]):
            p = pen[:, l]
            if l == pen.shape[1] - 1 and end_beat is not None:
                p = N.ones(pen.shape[0]) * N.inf
                p[end_beat] = pen[end_beat, -1]
            vals = p + tc + cost[:, l - 1][:, N.newaxis]
            min_nodes = __fast_argmin_axis_0(vals)
            min_vals = N.amin(vals, axis=0)
            cost[:, l] = min_vals
            prev_node[:, l] = min_nodes

        end = N.argmin(cost[:, -1])
        path = [end]
        node = end
        length = pen.shape[1] - 1
        while length > 0:
            node = prev_node[int(node), length]
            path.append(node)
            length -= 1

        path = [x for x in reversed(path)]
        return path


    def divide_and_conquer_cost_and_path(tc, pen, start_beat, end_beat, offset):
        l = pen.shape[1]  # out beats
        opt_path = []

        if l == 0: return
        elif l == 1: return
        elif l == 2 and start_beat is not None and end_beat is not None: return
        elif l == 2 and start_beat is not None:
            new_pen = N.empty(all_full)
            get_pen_column(pen, 1, new_pen, offset)

            tc_column = N.empty(all_full)
            if start_beat >= p0_full:
                # start beat is pause
                j = n_beats + (start_beat - p0_full)
                get_tc_column(tc, start_beat, tc_column, backward=True)
            else:
                beat_i_of_start = int(start_beat / float(n_beats))
                beat_j_of_start = start_beat % n_beats
                get_tc_column(tc, start_beat, tc_column, backward=True)

            global_path[offset] = start_beat
            global_path[offset + 1] = N.argmin(tc_column + new_pen)

            global_path_cost[offset + 1] = N.min(tc_column + new_pen)
            return
        elif l == 2 and end_beat is not None:
            new_pen = N.empty(all_full)
            get_pen_column(pen, 0, new_pen, offset)

            tc_column = N.empty(all_full)
            if end_beat >= p0_full:
                # end beat is pause
                j = n_beats + (end_beat - p0_full)
                get_tc_column(tc, end_beat, tc_column)
            else:
                beat_i_of_end = int(end_beat / float(n_beats))
                beat_j_of_end = end_beat % n_beats
                get_tc_column(tc, end_beat, tc_column)

            global_path[offset] = N.argmin(tc_column + new_pen)
            global_path[offset + 1] = end_beat

            global_path_cost[offset] = N.min(tc_column + new_pen)
            return
        elif l == 2:
            print "actually running full optimize"
            import pdb; pdb.set_trace()
            # opt_path = cost_and_path(tc, pen, start_beat, end_beat)
            # global_path[offset:offset + pen.shape[1]] = opt_path
            return

        l_over_2 = N.floor(l / 2.0)

        f = space_efficient_cost_with_duration_constraint(tc, pen[:, :l_over_2 + 1], start_beat, None, offset)
        g = backward_space_efficient_cost_with_duration_constraint(tc, pen[:, l_over_2:], None, end_beat, offset + l_over_2)

        # f = space_efficient_cost(tc, pen[:, :l_over_2 + 1], start_beat, None)
        # g = backward_space_efficient_cost(tc, pen[:, l_over_2:], None, end_beat)


        opt_i = N.argmin(f + g)
        global_path[l_over_2 + offset] = opt_i
        global_path_cost[l_over_2 + offset] = N.min(f)

        # first half
        divide_and_conquer_cost_and_path(
            tc, pen[:, :l_over_2 + 1], start_beat, opt_i, offset)

        # second half
        divide_and_conquer_cost_and_path(
            tc, pen[:, l_over_2:], opt_i, end_beat, l_over_2 + offset)

        return

    global_path_cost = N.zeros(penalty.shape[1])
    global_path = N.zeros(penalty.shape[1], dtype=N.int)
    divide_and_conquer_cost_and_path(trans_cost, penalty, None, None, 0)

    return global_path, global_path_cost