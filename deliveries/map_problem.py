from framework import *

from typing import Iterator, Optional, Callable
from dataclasses import dataclass


__all__ = ['MapState', 'MapProblem']


@dataclass(frozen=True)
class MapState(GraphProblemState):
    """
    StreetsMap state is represents the current geographic location on the map.
    This location is defined by the junction index.
    """
    junction_id: int

    def __eq__(self, other):
        assert isinstance(other, MapState)
        return other.junction_id == self.junction_id

    def __hash__(self):
        return hash(self.junction_id)

    def __str__(self):
        return str(self.junction_id).rjust(5, ' ')


class MapProblem(GraphProblem):
    """
    Represents a problem on the streets map.
    The problem is defined by a source location on the map and a destination.
    """

    name = 'StreetsMap'

    def __init__(self, streets_map: StreetsMap, source_junction_id: int, target_junction_id: int,
                 road_cost_fn: Optional[Callable[[Link], Cost]] = None,
                 zero_road_cost: Optional[Cost] = None):
        initial_state = MapState(source_junction_id)
        super(MapProblem, self).__init__(initial_state)
        self.streets_map = streets_map
        self.target_junction_id = target_junction_id
        self.road_cost_fn = road_cost_fn
        self.zero_road_cost = zero_road_cost
        self.name += f'(src: {source_junction_id} dst: {target_junction_id})'
    
    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[OperatorResult]:
        """
        For a given state, iterates over its successor states.
        The successor states represents the junctions to which there
         exists a road originates from the given state.
        """

        # All of the states in this problem are instances of the class `MapState`.
        assert isinstance(state_to_expand, MapState)

        # Get the junction (in the map) that is represented by the state to expand.
        junction = self.streets_map[state_to_expand.junction_id]

        for link in junction.outgoing_links:
            successor_state = MapState(link.target)
            if self.road_cost_fn is None:
                cost = link.distance
            else:
                cost = self.road_cost_fn(link)
            yield OperatorResult(successor_state, cost)

    def is_goal(self, state: GraphProblemState) -> bool:
        """
        :return: Whether a given map state represents the destination.
        """
        assert (isinstance(state, MapState))

        return state.junction_id == self.target_junction_id

    def get_zero_cost(self) -> Cost:
        if self.zero_road_cost is not None:
            return self.zero_road_cost
        return 0.0
