from framework import *
from deliveries import *

from matplotlib import pyplot as plt
import numpy as np
from typing import List, Union, Optional

# Load the streets map
streets_map = StreetsMap.load_from_csv(Consts.get_data_file_path("tlv_streets_map.csv"))

# Make sure that the whole execution is deterministic.
# This is important, because we expect to get the exact same results
# in each execution.
Consts.set_seed()


# --------------------------------------------------------------------
# ------------------------ StreetsMap Problem ------------------------
# --------------------------------------------------------------------

def plot_distance_and_expanded_wrt_weight_figure(
        problem_name: str,
        weights: Union[np.ndarray, List[float]],
        total_cost: Union[np.ndarray, List[float]],
        total_nr_expanded: Union[np.ndarray, List[int]]):
    """
    Use `matplotlib` to generate a figure of the distance & #expanded-nodes
     w.r.t. the weight.
    DONE [Ex.14]
    """
    weights, total_cost, total_nr_expanded = np.array(weights), np.array(total_cost), np.array(total_nr_expanded)
    assert len(weights) == len(total_cost) == len(total_nr_expanded)
    assert len(weights) > 0
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    assert is_sorted(weights)

    fig, ax1 = plt.subplots()

    p1, = ax1.plot(weights, total_cost, color="b", linestyle="solid")
    p1.set_label("Solution cost")

    # ax1: Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('solution cost', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel('weight')

    # Create another axis for the #expanded curve.
    ax2 = ax1.twinx()

    p2, = ax2.plot(weights, total_nr_expanded, color="r", linestyle="solid")
    p2.set_label("#Expanded states")

    curves = [p1, p2]
    ax1.legend(curves, [curve.get_label() for curve in curves])

    fig.tight_layout()
    plt.title(f'Quality vs. time for wA* \non problem {problem_name}')
    plt.show()


def run_astar_for_weights_in_range(heuristic_type: HeuristicFunctionType, problem: GraphProblem, n: int = 30,
                                   max_nr_states_to_expand: Optional[int] = 30_000):
    costs = []
    number_of_expanded_states = []
    weights = []

    examined_weights = np.linspace(start=0.5, stop=1, num=n)
    for weight in examined_weights:
        a_star = AStar(heuristic_type, weight, max_nr_states_to_expand)
        res = a_star.solve_problem(problem)
        if res.is_solution_found:
            costs.append(res.solution_g_cost)
            number_of_expanded_states.append(res.nr_expanded_states)
            weights.append(weight)

    plot_distance_and_expanded_wrt_weight_figure(problem_name=problem.name,
                                                 weights=weights,
                                                 total_cost=costs,
                                                 total_nr_expanded=number_of_expanded_states)


def toy_map_problem_experiments():
    print()
    print('Solve the map problem.')

    # Ex.10
    toy_map_problem = MapProblem(streets_map, 54, 549)
    uc = UniformCost()
    res = uc.solve_problem(toy_map_problem)
    print(res)

    # Ex.12
    a_star = AStar(NullHeuristic)
    res = a_star.solve_problem(toy_map_problem)
    print(res)

    # Ex.13
    a_star = AStar(AirDistHeuristic)
    res = a_star.solve_problem(toy_map_problem)
    print(res)

    # Ex.14
    # DONE:
    #  1. Complete the implementation of the function
    #     `run_astar_for_weights_in_range()` (upper in this file).
    #  2. Complete the implementation of the function
    #     `plot_distance_and_expanded_wrt_weight_figure()`
    #     (upper in this file).
    #  3. Call here the function `run_astar_for_weights_in_range()`
    #     with `AirDistHeuristic` and `toy_map_problem`.
    run_astar_for_weights_in_range(AirDistHeuristic, toy_map_problem)


# --------------------------------------------------------------------
# --------------------- Truck Deliveries Problem ---------------------
# --------------------------------------------------------------------

loaded_problem_inputs_by_size = {}
loaded_problems_by_size_and_opt_obj = {}


def get_deliveries_problem(problem_input_size: str = 'small',
                           optimization_objective: OptimizationObjective = OptimizationObjective.Distance):
    if (problem_input_size, optimization_objective) in loaded_problems_by_size_and_opt_obj:
        return loaded_problems_by_size_and_opt_obj[(problem_input_size, optimization_objective)]
    assert problem_input_size in {'small', 'moderate', 'big'}
    if problem_input_size not in loaded_problem_inputs_by_size:
        loaded_problem_inputs_by_size[problem_input_size] = DeliveriesTruckProblemInput.load_from_file(
            f'{problem_input_size}_delivery.in', streets_map)
    problem = DeliveriesTruckProblem(
        problem_input=loaded_problem_inputs_by_size[problem_input_size],
        streets_map=streets_map,
        optimization_objective=optimization_objective)
    loaded_problems_by_size_and_opt_obj[(problem_input_size, optimization_objective)] = problem
    return problem


def basic_deliveries_truck_problem_experiments():
    print()
    print('Solve the truck deliveries problem (small input, only distance objective, UniformCost).')

    small_delivery_problem_with_distance_cost = get_deliveries_problem('small', OptimizationObjective.Distance)

    # Ex.16
    # DONE: create an instance of `UniformCost`, solve the `small_delivery_problem_with_distance_cost`
    #       with it and print the results.
    uc = UniformCost()
    res = uc.solve_problem(small_delivery_problem_with_distance_cost)
    print(res)


def deliveries_truck_problem_with_astar_experiments():
    print()
    print('Solve the truck deliveries problem (moderate input, only distance objective, A*, MaxAirDist & SumAirDist & MSTAirDist heuristics).')

    moderate_delivery_problem_with_distance_cost = get_deliveries_problem('moderate', OptimizationObjective.Distance)

    # Ex.18
    # DONE: create an instance of `AStar` with the `TruckDeliveriesMaxAirDistHeuristic`,
    #       solve the `moderate_delivery_problem_with_distance_cost` with it and print the results.
    a_star = AStar(TruckDeliveriesMaxAirDistHeuristic)
    res = a_star.solve_problem(moderate_delivery_problem_with_distance_cost)
    print(res)

    # Ex.21
    # DONE: create an instance of `AStar` with the `TruckDeliveriesSumAirDistHeuristic`,
    #       solve the `moderate_delivery_problem_with_distance_cost` with it and print the results.
    a_star = AStar(TruckDeliveriesSumAirDistHeuristic)
    res = a_star.solve_problem(moderate_delivery_problem_with_distance_cost)
    print(res)

    # Ex.24
    # DONE: create an instance of `AStar` with the `TruckDeliveriesMSTAirDistHeuristic`,
    #       solve the `moderate_delivery_problem_with_distance_cost` with it and print the results.
    a_star = AStar(TruckDeliveriesMSTAirDistHeuristic)
    res = a_star.solve_problem(moderate_delivery_problem_with_distance_cost)
    print(res)


def deliveries_truck_problem_with_weighted_astar_experiments():
    print()
    print('Solve the truck deliveries problem (small & moderate input, only distance objective, wA*).')

    small_delivery_problem_with_distance_cost = get_deliveries_problem('small', OptimizationObjective.Distance)
    moderate_delivery_problem_with_distance_cost = get_deliveries_problem('moderate', OptimizationObjective.Distance)

    # Ex.26
    # DONE: Call here the function `run_astar_for_weights_in_range()`
    #       with `TruckDeliveriesMSTAirDistHeuristic`
    #       over the `small_delivery_problem_with_distance_cost`.
    run_astar_for_weights_in_range(TruckDeliveriesMSTAirDistHeuristic, small_delivery_problem_with_distance_cost)

    # Ex.26
    # DONE: Call here the function `run_astar_for_weights_in_range()`
    #       with `TruckDeliveriesSumAirDistHeuristic`
    #       over the `moderate_delivery_problem_with_distance_cost`.
    run_astar_for_weights_in_range(TruckDeliveriesSumAirDistHeuristic, moderate_delivery_problem_with_distance_cost)


def multiple_objectives_deliveries_truck_problem_experiments():
    print()
    print('Solve the truck deliveries problem (small input, time & money objectives).')

    small_delivery_problem_with_time_cost = get_deliveries_problem('small', OptimizationObjective.Time)
    small_delivery_problem_with_money_cost = get_deliveries_problem('small', OptimizationObjective.Money)

    # Ex.29
    # DONE: create an instance of `AStar` with the `TruckDeliveriesMSTAirDistHeuristic`,
    #       solve the `small_delivery_problem_with_time_cost` with it and print the results.
    a_star = AStar(TruckDeliveriesMSTAirDistHeuristic)
    res = a_star.solve_problem(small_delivery_problem_with_time_cost)
    print(res)

    # Ex.29
    # DONE: create an instance of `AStar` with the `TruckDeliveriesMSTAirDistHeuristic`,
    #       solve the `small_delivery_problem_with_money_cost` with it and print the results.
    a_star = AStar(TruckDeliveriesMSTAirDistHeuristic)
    res = a_star.solve_problem(small_delivery_problem_with_money_cost)
    print(res)


def deliveries_truck_problem_with_astar_epsilon_experiments():
    print()
    print('Solve the truck deliveries problem (moderate input, distance objective, using A*eps, use non-acceptable '
          'heuristic as focal heuristic).')

    moderate_delivery_problem_with_distance_cost = get_deliveries_problem('moderate', OptimizationObjective.Distance)

    # Firstly solve the problem with AStar & MST heuristic for having a reference for #devs.
    a_star = AStar(TruckDeliveriesMSTAirDistHeuristic)
    res = a_star.solve_problem(moderate_delivery_problem_with_distance_cost)
    print(res)

    def within_focal_h_sum_priority_function(node: SearchNode, problem: GraphProblem, solver: AStarEpsilon):
        if not hasattr(solver, '__focal_heuristic'):
            setattr(solver, '__focal_heuristic', TruckDeliveriesSumAirDistHeuristic(problem=problem))
        focal_heuristic = getattr(solver, '__focal_heuristic')
        return focal_heuristic.estimate(node.state)

    # Ex.33
    # Try using A*eps to improve the speed (#dev) with a non-acceptable heuristic.
    # DONE: create an instance of `AStarEpsilon` with the `TruckDeliveriesMSTAirDistHeuristic`,
    #       solve the `moderate_delivery_problem_with_distance_cost` with it and print the results.
    #       use focal_epsilon=0.03, and  max_focal_size=40.
    #       use within_focal_priority_function=within_focal_h_sum_priority_function
    a_star_eps = AStarEpsilon(heuristic_function_type=TruckDeliveriesMSTAirDistHeuristic,
                              within_focal_priority_function=within_focal_h_sum_priority_function,
                              focal_epsilon=0.03, max_focal_size=40)
    res = a_star_eps.solve_problem(moderate_delivery_problem_with_distance_cost)
    print(res)


def deliveries_truck_problem_anytime_astar_experiments():
    print()
    print('Solve the truck deliveries problem (moderate input, only distance objective, Anytime-A*, '
          'MSTAirDist heuristics).')

    moderate_delivery_problem_with_distance_cost = get_deliveries_problem('moderate', OptimizationObjective.Distance)

    # Ex.35
    # DONE: create an instance of `AnytimeAStar` once with the `TruckDeliveriesMSTAirDistHeuristic`, with
    #       `max_nr_states_to_expand_per_iteration` set to 50, solve the
    #       `moderate_delivery_problem_with_distance_cost` with it and print the results.
    anytime_a_star = AnytimeAStar(heuristic_function_type=TruckDeliveriesMSTAirDistHeuristic,
                                  max_nr_states_to_expand_per_iteration=50)
    res = anytime_a_star.solve_problem(moderate_delivery_problem_with_distance_cost)
    print(res)


def big_deliveries_truck_problem_with_non_acceptable_heuristic_and_anytime_astar_experiments():
    print()
    print('Solve the truck deliveries problem (big input, only distance objective, Anytime-A*, '
          'SumAirDist & MSTAirDist heuristics).')

    big_delivery_problem_with_distance_cost = get_deliveries_problem('big', OptimizationObjective.Distance)

    # Ex.35
    # DONE: create an instance of `AnytimeAStar` once with the `TruckDeliveriesSumAirDistHeuristic`,
    #       and then with the `TruckDeliveriesMSTAirDistHeuristic`, both with `max_nr_states_to_expand_per_iteration`
    #       set to 400, solve the `big_delivery_problem_with_distance_cost` with it and print the results.
    anytime_a_star = AnytimeAStar(heuristic_function_type=TruckDeliveriesSumAirDistHeuristic,
                                  max_nr_states_to_expand_per_iteration=400)
    res = anytime_a_star.solve_problem(big_delivery_problem_with_distance_cost)
    print(res)

    anytime_a_star = AnytimeAStar(heuristic_function_type=TruckDeliveriesMSTAirDistHeuristic,
                                  max_nr_states_to_expand_per_iteration=400)
    res = anytime_a_star.solve_problem(big_delivery_problem_with_distance_cost)
    print(res)


def run_all_experiments():
    toy_map_problem_experiments()
    basic_deliveries_truck_problem_experiments()
    deliveries_truck_problem_with_astar_experiments()
    deliveries_truck_problem_with_weighted_astar_experiments()
    multiple_objectives_deliveries_truck_problem_experiments()
    deliveries_truck_problem_with_astar_epsilon_experiments()
    deliveries_truck_problem_anytime_astar_experiments()
    big_deliveries_truck_problem_with_non_acceptable_heuristic_and_anytime_astar_experiments()


if __name__ == '__main__':
    run_all_experiments()
