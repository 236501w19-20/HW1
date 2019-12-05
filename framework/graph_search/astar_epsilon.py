from .graph_problem_interface import *
from .astar import AStar
from typing import Optional, Callable
import numpy as np
import math


class AStarEpsilon(AStar):
    """
    This class implements the (weighted) A*Epsilon search algorithm.
    A*Epsilon algorithm basically works like the A* algorithm, but with
    another way to choose the next node to expand from the open queue.
    """

    solver_name = 'A*eps'

    def __init__(self,
                 heuristic_function_type: HeuristicFunctionType,
                 within_focal_priority_function: Callable[[SearchNode, GraphProblem, 'AStarEpsilon'], float],
                 heuristic_weight: float = 0.5,
                 max_nr_states_to_expand: Optional[int] = None,
                 focal_epsilon: float = 0.1,
                 max_focal_size: Optional[int] = None):
        # A* is a graph search algorithm. Hence, we use close set.
        super(AStarEpsilon, self).__init__(heuristic_function_type, heuristic_weight,
                                           max_nr_states_to_expand=max_nr_states_to_expand)
        self.focal_epsilon = focal_epsilon
        if focal_epsilon < 0:
            raise ValueError(f'The argument `focal_epsilon` for A*eps should be >= 0; '
                             f'given focal_epsilon={focal_epsilon}.')
        self.within_focal_priority_function = within_focal_priority_function
        self.max_focal_size = max_focal_size

    def _init_solver(self, problem):
        super(AStarEpsilon, self)._init_solver(problem)

    def _extract_next_search_node_to_expand(self, problem: GraphProblem) -> Optional[SearchNode]:
        """
        Extracts the next node to expand from the open queue,
         by focusing on the current FOCAL and choosing the node
         with the best within_focal_priority from it.
        DONE [Ex.32]: Implement this method!
        Find the minimum expanding-priority value in the `open` queue.
        Calculate the maximum expanding-priority of the FOCAL, which is
         the min expanding-priority in open multiplied by (1 + eps) where
         eps is stored under `self.focal_epsilon`.
        Create the FOCAL by popping items from the `open` queue and inserting
         them into a focal list. Don't forget to satisfy the constraint of
         `self.max_focal_size` if it is set (not None). #Typo - in FAQ.
        Notice: You might want to pop items from the `open` priority queue,
         and then choose an item out of these popped items. Don't forget:
         the other items have to be pushed back into open.
        Inspect the base class `BestFirstSearch` to retrieve the type of
         the field `open`. Then find the definition of this type and find
         the right methods to use (you might want to peek the head node, to
         pop/push nodes and to query whether the queue is empty).
        For each node (candidate) in the created focal, calculate its priority
         by calling the function `self.within_focal_priority_function` on it.
         This function expects to get 3 values: the node, the problem and the
         solver (self). You can create an array of these priority value. Then,
         use `np.argmin()` to find the index of the item (within this array)
         with the minimal value. After having this index you could pop this
         item from the focal (at this index). This is the node that have to
         be eventually returned.
        Don't forget to handle correctly corner-case like when the open queue
         is empty. In this case a value of `None` has to be returned.
        Note: All the nodes that were in the open queue at the beginning of this
         method should be kept in the open queue at the end of this method, except
         for the extracted (and returned) node.
        """
        if self.open.is_empty():
            return None

        # In priority queue peek or pop will return the node with lowest priority
        min_expanding_priority = self.open.peek_next_node().expanding_priority
        max_focal_expanding_priority = (1 + self.focal_epsilon) * min_expanding_priority

        # If not given the max focal size, it's about of open's size
        real_max_focal_size = len(self.open)
        if self.max_focal_size is not None:
            real_max_focal_size = self.max_focal_size

        # Select nodes to focal
        focal = []
        while (not self.open.is_empty()) and len(focal) < real_max_focal_size:
            next_node = self.open.peek_next_node()
            if next_node.expanding_priority <= max_focal_expanding_priority:
                focal.append(self.open.pop_next_node())
            else:
                break

        # Choose best from focal
        focal_priorities = [self.within_focal_priority_function(item, problem, self) for item in focal]

        # Only the first occurrence is returned, and can be converted to int
        min_focal_priority_index = np.argmin(focal_priorities)

        node_to_return = focal.pop(int(min_focal_priority_index))

        # Return back to queue others from focal
        for node_to_push_back in focal:
            self.open.push_node(node_to_push_back)

        # As in BestFirstSearch and AStar, we need to add it to close
        if self.use_close:
            self.close.add_node(node_to_return)

        return node_to_return
