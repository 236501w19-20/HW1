from typing import *
from dataclasses import dataclass

from framework import *
from .map_problem import MapProblem, MapState
from .cached_map_distance_finder import CachedMapDistanceFinder
from .deliveries_truck_problem_input import *


__all__ = ['DeliveriesTruckState', 'DeliveryCost', 'DeliveriesTruckProblem', 'TruckDeliveriesInnerMapProblemHeuristic']


@dataclass(frozen=True)
class DeliveriesTruckState(GraphProblemState):
    """
    An instance of this class represents a state of deliveries problem.
    This state includes the deliveries which are currently loaded on the
     truck, the deliveries which had already been dropped, and the current
     location of the truck (which is either the initial location or the
     last pick/drop location.
    """

    loaded_deliveries: FrozenSet[Delivery]
    dropped_deliveries: FrozenSet[Delivery]
    current_location: Junction

    def get_last_action(self) -> Optional[Tuple[str, Delivery]]:
        found_loaded_delivery = next((loaded_delivery for loaded_delivery in self.loaded_deliveries
                                      if loaded_delivery.pick_location == self.current_location), None)
        if found_loaded_delivery is not None:
            return 'pick', found_loaded_delivery

        found_dropped_delivery = next((dropped_delivery for dropped_delivery in self.dropped_deliveries
                                       if dropped_delivery.drop_location == self.current_location), None)
        if found_dropped_delivery is not None:
            return 'drop', found_dropped_delivery

        return None

    def __str__(self):
        last_action = self.get_last_action()
        current_location = 'initial-location' if last_action is None else f'{last_action[0]} loc @ {last_action[1]}'

        return f'(dropped: {list(self.dropped_deliveries)} ' \
               f'loaded: {list(self.loaded_deliveries)} ' \
               f'current_location: {current_location})'

    def __eq__(self, other):
        """
        This method is used to determine whether two given state objects represent the same state.
        """
        assert isinstance(other, DeliveriesTruckState)

        # TODO [Ex.15]: Complete the implementation of this method!
        #  Note that you can simply compare two instances of `Junction` type
        #   (using equals `==` operator) because the class `Junction` explicitly
        #   implements the `__eq__()` method. The types `frozenset` and `Delivery`
        #   are also comparable (in the same manner).
        raise NotImplementedError()  # TODO: remove this line.

    def __hash__(self):
        """
        This method is used to create a hash of a state instance.
        The hash of a state being is used whenever the state is stored as a key in a dictionary
         or as an item in a set.
        It is critical that two objects representing the same state would have the same hash!
        """
        return hash((self.loaded_deliveries, self.dropped_deliveries, self.current_location))

    def get_total_nr_packages_loaded(self) -> int:
        """
        This method returns the total number of packages that are loaded on the truck in this state.
        TODO [Ex.15]: Implement this method.
         Notice that this method can be implemented using a single line of code - do so!
         Use python's built-it `sum()` function.
         Notice that `sum()` can receive an *ITERATOR* as argument; That is, you can simply write something like this:
        >>> sum(<some expression using item> for item in some_collection_of_items)
        """
        raise NotImplementedError()  # TODO: remove this line.


@dataclass(frozen=True)
class DeliveryCost(ExtendedCost):
    """
    An instance of this class is returned as an operator cost by the method
     `DeliveriesTruckProblem.expand_state_with_costs()`.
    The `SearchNode`s that will be created during the run of the search algorithm are going
     to have instances of `DeliveryCost` in SearchNode's `cost` field (instead of float value).
    The reason for using a custom type for the cost (instead of just using a `float` scalar),
     is because we want the cumulative cost (of each search node and particularly of the final
     node of the solution) to be consisted of 3 objectives: (i) distance, (ii) time, and
     (iii) money.
    The field `optimization_objective` controls the objective of the problem (the cost we want
     the solver to minimize). In order to tell the solver which is the objective to optimize,
     we have the `get_g_cost()` method, which returns a single `float` scalar which is only the
     cost to optimize.
    This way, whenever we get a solution, we can inspect the 3 different costs of that solution,
     even though the objective was only one of the costs (time for example).
    Having said that, note that during this assignment we will mostly use the distance objective.
    """
    distance_cost: float = 0.0
    time_cost: float = 0.0
    money_cost: float = 0.0
    optimization_objective: OptimizationObjective = OptimizationObjective.Distance

    def __add__(self, other):
        assert isinstance(other, DeliveryCost)
        assert other.optimization_objective == self.optimization_objective
        return DeliveryCost(optimization_objective=self.optimization_objective,
                            distance_cost=self.distance_cost + other.distance_cost,
                            time_cost=self.time_cost + other.time_cost,
                            money_cost=self.money_cost + other.money_cost)

    def get_g_cost(self) -> float:
        if self.optimization_objective == OptimizationObjective.Distance:
            return self.distance_cost
        elif self.optimization_objective == OptimizationObjective.Time:
            return self.time_cost
        else:
            assert self.optimization_objective == OptimizationObjective.Money
            return self.money_cost

    def __repr__(self):
        return f'DeliveryCost(' \
               f'dist={self.distance_cost:11.3f} meter, ' \
               f'time={self.time_cost:11.3f} minutes, ' \
               f'money={self.money_cost:11.3f} nis)'


class DeliveriesTruckProblem(GraphProblem):
    """
    An instance of this class represents a deliveries truck problem.
    """

    name = 'Deliveries'

    def __init__(self,
                 problem_input: DeliveriesTruckProblemInput,
                 streets_map: StreetsMap,
                 optimization_objective: OptimizationObjective = OptimizationObjective.Distance):
        self.name += f'({problem_input.input_name}({len(problem_input.deliveries)}):{optimization_objective.name})'
        initial_state = DeliveriesTruckState(
            loaded_deliveries=frozenset(),
            dropped_deliveries=frozenset(),
            current_location=problem_input.delivery_truck.initial_location
        )
        super(DeliveriesTruckProblem, self).__init__(initial_state)
        self.problem_input = problem_input
        self.streets_map = streets_map
        inner_map_problem_heuristic_type = lambda problem: TruckDeliveriesInnerMapProblemHeuristic(problem, self)
        self.map_distance_finder = CachedMapDistanceFinder(
            streets_map, AStar(inner_map_problem_heuristic_type),
            road_cost_fn=self._calc_map_road_cost,
            zero_road_cost=DeliveryCost(optimization_objective=optimization_objective))
        self.optimization_objective = optimization_objective

    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[OperatorResult]:
        """
        TODO [Ex.15]: Implement this method!
        This method represents the `Succ: S -> P(S)` function of the deliveries truck problem.
        The `Succ` function is defined by the problem operators as shown in class.
        The deliveries truck problem operators are defined in the assignment instructions.
        It receives a state and iterates over its successor states.
        Notice that this its return type is an *Iterator*. It means that this function is not
         a regular function, but a `generator function`. Hence, it should be implemented using
         the `yield` statement.
        For each successor, an object of type `OperatorResult` is yielded. This object describes the
            successor state, the cost of the applied operator and its name. Look for its definition
            and use the correct fields in its c'tor. The operator name should be in the following
            format: `pick ClientName` (with the correct client name) if a pick operator was applied,
            or `drop ClientName` if a drop operator was applied. The delivery object stores its
            client name in one of its fields.
        Things you might want to use:
            - The method `self.get_total_nr_packages_loaded()`.
            - The field `self.problem_input.delivery_truck.max_nr_loaded_packages`.
            - The method `self.get_deliveries_waiting_to_pick()` here.
            - The method `self.map_distance_finder.get_map_cost_between()` to calculate
              the operator cost. Its returned value is the operator cost (as is).
            - The c'tor for `DeliveriesTruckState` to create the new successor state.
            - Python's built-in method `frozenset()` to create a new frozen set (for fields that
              expect this type).
            - Other fields of the state and the problem input.
        """

        assert isinstance(state_to_expand, DeliveriesTruckState)
        raise NotImplementedError()  # TODO: remove this line!

    def is_goal(self, state: GraphProblemState) -> bool:
        """
        This method receives a state and returns whether this state is a goal.
        TODO [Ex.15]: implement this method!
        """
        assert isinstance(state, DeliveriesTruckState)
        raise NotImplementedError()  # TODO: remove the line!

    def _calc_map_road_cost(self, link: Link) -> DeliveryCost:
        """
        TODO [Ex.27]: Modify the implementation of this method, so that for a given link (road), it would return
                the extended cost of this link. That is, the distance should remain as it is now, but both
                the `time_cost` and the `money_cost` should be set appropriately.
            Use the `optimal_velocity` and the `gas_cost_per_meter` returned by the method
                `self.problem_input.delivery_truck.calc_optimal_driving_parameters()`, in order to calculate
                both the `time_cost` and the `money_cost`.
            Note that the `money_cost` is the total gas cost for this given link plus the total fee paid
                for driving on this road if this road is a toll road. Use the appropriate Link's field to
                check whether is it a toll road and to get the distance of this road. Additionally, use the
                appropriate field in the problem input (accessible by `self.problem_input`) to get the toll road
                cost per meter.
        """
        optimal_velocity, gas_cost_per_meter = self.problem_input.delivery_truck.calc_optimal_driving_parameters(
            optimization_objective=self.optimization_objective, max_driving_speed=link.max_speed)
        return DeliveryCost(
            distance_cost=link.distance,
            time_cost=0,  # TODO: modify this value!
            money_cost=0,  # TODO: modify this value!
            optimization_objective=self.optimization_objective)

    def get_zero_cost(self) -> Cost:
        return DeliveryCost(optimization_objective=self.optimization_objective)

    def get_cost_lower_bound_from_distance_lower_bound(self, total_distance_lower_bound: float) -> float:
        """
        Used by the heuristics of the deliveries truck problem.
        Given a lower bound of the distance (in meters) that the truck has left to travel,
         this method returns an appropriate lower bound of the distance/time/money cost
         based on the problem's objective.
        TODO [Ex.28]: We left only partial implementation of this method (just the trivial distance objective).
            Complete the implementation of this method!
            You might want to use constants like `MIN_ROAD_SPEED` or `MAX_ROAD_SPEED`.
            For the money cost, you would like to use the method `self._calc_map_road_cost()`. This
                method expects to get a `Link` instance and returns the (extended) cost of this road.
                You'll have to take the money cost from this extended cost.
            Although the `total_distance_lower_bound` actually represents an estimation for the
                remaining route (and not an actual road on the map), you can simply create a `Link`
                instance (that represents this whole remaining path) for this purpose.
            Remember: The return value should be a real lower bound. This is required for the
                heuristic to be acceptable.
        """
        if self.optimization_objective == OptimizationObjective.Distance:
            return total_distance_lower_bound
        elif self.optimization_objective == OptimizationObjective.Time:
            raise NotImplementedError()  # TODO: remove this line and complete the implementation of this case!
        else:
            assert self.optimization_objective == OptimizationObjective.Money
            raise NotImplementedError()  # TODO: remove this line and complete the implementation of this case!

    def get_deliveries_waiting_to_pick(self, state: DeliveriesTruckState) -> Set[Delivery]:
        """
        This method returns a set of all deliveries that haven't been neither picked nor dropped yet.
        TODO [Ex.15]: Implement this method.
            Use `set` difference operations.
            Note: Given a collection of items, you can create a new set of these items simply by
                `set(my_collection_of_items)`. Then you can use set operations over this newly
                generated set.
            Note: This method can be implemented using a single line of code.
        """
        raise NotImplementedError()  # TODO: remove this line!

    def get_all_junctions_in_remaining_truck_path(self, state: DeliveriesTruckState) -> Set[Junction]:
        """
        This method returns a set of all junctions that are part of the remaining route of the truck.
        This includes the truck's current location, the pick locations of the deliveries that haven't
         been picked yet, and the drop location of the deliveries that haven't been dropped yet.
        TODO [Ex.17]: Implement this method.
            Use `set` union operations.
            Use the method `self.get_deliveries_waiting_to_pick(state)`.
            Note: `set-comprehension` technique might be useful here. It works similar to the
                `list-comprehension` technique. Example: {i * 10 for i in range(100)} would create
                a set of the items {0, 10, 20, 30, ..., 990}
        """
        raise NotImplementedError()  # TODO: remove this line!


class TruckDeliveriesInnerMapProblemHeuristic(HeuristicFunction):
    heuristic_name = 'DeliveriesCostBasedOnAirDist'

    def __init__(self, inner_map_problem: GraphProblem, outer_deliveries_problem: DeliveriesTruckProblem):
        super(TruckDeliveriesInnerMapProblemHeuristic, self).__init__(inner_map_problem)
        assert isinstance(self.problem, MapProblem)
        self.outer_deliveries_problem = outer_deliveries_problem

    def estimate(self, state: GraphProblemState) -> float:
        assert isinstance(self.problem, MapProblem)
        assert isinstance(state, MapState)

        source_junction = self.problem.streets_map[state.junction_id]
        target_junction = self.problem.streets_map[self.problem.target_junction_id]
        total_distance_lower_bound = source_junction.calc_air_distance_from(target_junction)
        return self.outer_deliveries_problem.get_cost_lower_bound_from_distance_lower_bound(total_distance_lower_bound)
