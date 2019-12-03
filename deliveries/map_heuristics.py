from framework.graph_search import *
from .map_problem import MapProblem, MapState


class AirDistHeuristic(HeuristicFunction):
    heuristic_name = 'AirDist'

    def estimate(self, state: GraphProblemState) -> float:
        """
        The air distance between the geographic location represented
         by `state` and the geographic location of the problem's target.
        """
        assert isinstance(self.problem, MapProblem)
        assert isinstance(state, MapState)

        street_map = self.problem.streets_map
        current_junction = street_map[state.junction_id]
        target_junction = street_map[self.problem.target_junction_id]

        return target_junction.calc_air_distance_from(current_junction)
