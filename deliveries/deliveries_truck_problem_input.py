import os
from typing import *
from dataclasses import dataclass
from enum import Enum

from framework import *


__all__ = [
    'OptimizationObjective', 'Delivery', 'DeliveriesTruck', 'DeliveriesTruckProblemInput'
]


# class Serializable:
#     def serialize(self) -> str:
#         return ','.join(
#             getattr(self, field.name).serialize() if issubclass(field.type, Serializable) else str(getattr(self, field.name))
#             for field in fields(self)
#         )
#
#     @classmethod
#     def deserialize(cls, serialized: str, **kwargs) -> 'DeliveriesTruck':
#         parts = serialized.split(',')
#         return DeliveriesTruck(**{
#             field.name:
#                 field.type(getattr(self, field.name)) if issubclass(field.type, Serializable) else str(getattr(self, field.name))
#             for field in fields(cls)
#         })


class OptimizationObjective(Enum):
    Distance = 'Distance'
    Time = 'Time'
    Money = 'Money'


@dataclass(frozen=True)
class Delivery:
    delivery_id: int
    client_name: str
    pick_location: Junction
    drop_location: Junction
    nr_packages: int

    def serialize(self) -> str:
        return f'{self.delivery_id},{self.client_name},{self.pick_location.index},{self.drop_location.index},' \
               f'{self.nr_packages}'

    @staticmethod
    def deserialize(serialized: str, streets_map: StreetsMap) -> 'Delivery':
        parts = serialized.split(',')
        return Delivery(
            delivery_id=int(parts[0]),
            client_name=parts[1],
            pick_location=streets_map[int(parts[2])],
            drop_location=streets_map[int(parts[3])],
            nr_packages=int(parts[4]))

    def __repr__(self):
        return f'{self.client_name} ({self.nr_packages} pkgs)'

    def __hash__(self):
        return hash((self.delivery_id, self.pick_location, self.drop_location, self.nr_packages))


@dataclass(frozen=True)
class DeliveriesTruck:
    max_nr_loaded_packages: int
    initial_location: Junction
    optimal_vehicle_speed: float = kmph_to_mpm(87)
    gas_cost_per_meter_in_optimal_speed: float = 0.0009
    gas_cost_per_meter_gradient_wrt_speed_change: float = 0.0018

    def serialize(self) -> str:
        return f'{self.max_nr_loaded_packages},{self.initial_location.index},{self.optimal_vehicle_speed},' \
               f'{self.gas_cost_per_meter_in_optimal_speed},{self.gas_cost_per_meter_gradient_wrt_speed_change}'

    @staticmethod
    def deserialize(serialized: str, streets_map: StreetsMap) -> 'DeliveriesTruck':
        parts = serialized.split(',')
        assert len(parts) == 5
        return DeliveriesTruck(
            max_nr_loaded_packages=int(parts[0]),
            initial_location=streets_map[int(parts[1])],
            optimal_vehicle_speed=float(parts[2]),
            gas_cost_per_meter_in_optimal_speed=float(parts[3]),
            gas_cost_per_meter_gradient_wrt_speed_change=float(parts[4]))

    def calc_optimal_driving_parameters(self, optimization_objective: OptimizationObjective, max_driving_speed: float) \
            -> Tuple[float, float]:
        if optimization_objective == OptimizationObjective.Time or optimization_objective == OptimizationObjective.Distance:
            optimal_driving_speed = max_driving_speed
        else:
            assert optimization_objective == OptimizationObjective.Money
            optimal_driving_speed = self.optimal_vehicle_speed if self.optimal_vehicle_speed < max_driving_speed else max_driving_speed
        speed_delta_from_vehicle_optimal_speed = abs(optimal_driving_speed - self.optimal_vehicle_speed)
        max_speed_delta_from_vehicle_optimal_speed = max(abs(self.optimal_vehicle_speed - MIN_ROAD_SPEED), abs(MAX_ROAD_SPEED - self.optimal_vehicle_speed))
        relative_speed_delta_from_vehicle_optimal_speed = speed_delta_from_vehicle_optimal_speed / max_speed_delta_from_vehicle_optimal_speed
        gas_cost_per_meter = self.gas_cost_per_meter_in_optimal_speed + \
                             self.gas_cost_per_meter_gradient_wrt_speed_change * relative_speed_delta_from_vehicle_optimal_speed
        return optimal_driving_speed, gas_cost_per_meter


@dataclass(frozen=True)
class DeliveriesTruckProblemInput:
    input_name: str
    deliveries: Tuple[Delivery, ...]
    delivery_truck: DeliveriesTruck
    toll_road_cost_per_meter: float

    @staticmethod
    def load_from_file(input_file_name: str, streets_map: StreetsMap) -> 'DeliveriesTruckProblemInput':
        """
        Loads and parses a deliveries-problem-input from a file. Usage example:
        >>> problem_input = DeliveriesTruckProblemInput.load_from_file('big_delivery.in', streets_map)
        """

        with open(Consts.get_data_file_path(input_file_name), 'r') as input_file:
            input_type = input_file.readline().strip()
            if input_type != 'DeliveriesTruckProblemInput':
                raise ValueError(f'Input file `{input_file_name}` is not a deliveries input.')
            try:
                input_name = input_file.readline().strip()
                deliveries = tuple(
                    Delivery.deserialize(serialized_delivery, streets_map)
                    for serialized_delivery in input_file.readline().rstrip('\n').split(';'))
                delivery_truck = DeliveriesTruck.deserialize(input_file.readline().rstrip('\n'), streets_map)
                toll_road_cost_per_meter = float(input_file.readline())
            except:
                raise ValueError(f'Invalid input file `{input_file_name}`.')
        return DeliveriesTruckProblemInput(input_name=input_name, deliveries=deliveries, delivery_truck=delivery_truck,
                                           toll_road_cost_per_meter=toll_road_cost_per_meter)

    def store_to_file(self, input_file_name: str):
        with open(Consts.get_data_file_path(input_file_name), 'w') as input_file:
            lines = [
                'DeliveriesTruckProblemInput',
                str(self.input_name.strip()),
                ';'.join(delivery.serialize() for delivery in self.deliveries),
                self.delivery_truck.serialize(),
                str(self.toll_road_cost_per_meter)
            ]
            for line in lines:
                input_file.write(line + '\n')

    @staticmethod
    def load_all_inputs(streets_map: StreetsMap) -> Dict[str, 'DeliveriesTruckProblemInput']:
        """
        Loads all the inputs in the inputs directory.
        :return: list of inputs.
        """
        inputs = {}
        input_file_names = [f for f in os.listdir(Consts.DATA_PATH)
                            if os.path.isfile(os.path.join(Consts.DATA_PATH, f)) and f.split('.')[-1] == 'in']
        for input_file_name in input_file_names:
            try:
                problem_input = DeliveriesTruckProblemInput.load_from_file(input_file_name, streets_map)
                inputs[problem_input.input_name] = problem_input
            except:
                pass
        return inputs
