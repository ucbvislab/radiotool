subroutine build_table(transition_cost, penalty, cost, prev_node, n_beats, n_length)
    implicit none
    integer, intent(in) :: n_beats, n_length
    real(8), intent(in), dimension(0:n_beats - 1, 0:n_beats - 1) :: transition_cost
    real(8), intent(in), dimension(0:n_beats - 1, 0:n_length - 1) :: penalty
    real(8), intent(out), dimension(0:n_beats - 1, 0:n_length - 1) :: cost
    integer, intent(out), dimension(0:n_beats - 1, 0:n_length - 1) :: prev_node
    real(8), dimension(0:n_beats - 1) :: temp_cost
    real(8) :: min_val
    integer :: l, i, j, min_i
    cost(:, 0) = penalty(:, 0)
    do l = 1, n_length - 1
        do i = 0, n_beats - 1
            min_val = penalty(i, l) + transition_cost(0, i) + cost(0, l - 1)
            min_i = 0
            do j = 1,  n_beats - 1
                temp_cost(j) = penalty(i, l) + transition_cost(j, i) + cost(j, l - 1)
                if (temp_cost(j) < min_val) then
                    min_val = temp_cost(j)
                    min_i = j
                end if
            enddo
            cost(i, l) = min_val
            prev_node(i, l) = min_i
        enddo
    enddo
end subroutine