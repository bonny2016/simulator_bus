import random
import gym
import os
import gym.spaces
from gym.utils import seeding
from simulator.globals import N_STOPS, N_BUSES, CAPACITY, MAX_ONBOARD_PER_MINUTE, DEFAULT_BUS_SPEED
from enum import Enum
import numpy as np
from simulator.stop import BusStops
import pandas as pd
import torch
import networkx as nx
import matplotlib.pyplot as plt

stops = BusStops()

class BusStatus(Enum):
    CRUISING = 1
    HOLDING = 2
    ARRIVING = 3
    DECISION = 4 # after alighting & boarding are completed when arriving to a stop
class Bus:
    def __init__(self, id, status, holding_time,
                 speed=DEFAULT_BUS_SPEED, last_stop_idx=0, current_distance=0,
                 dispatch_time=-1,
                 passengers_on_board=np.zeros((N_STOPS, N_STOPS)),
                 passengers_travel_time=np.zeros((N_STOPS, N_STOPS))):
        self.id = id
        self.speed = speed
        self.dispatch_time = dispatch_time
        self.last_stop_idx = last_stop_idx
        self.last_stop = stops.get_stop_from_idx(last_stop_idx)
        self.current_distance = current_distance
        self.status = status
        self.holding_time = holding_time

        self.passengers_on_board_od = np.array(passengers_on_board)
        self.passengers_travel_time = np.array(passengers_travel_time)

        self.total_travel_time = 0
        self.time_since_last_stop = 0

    @property
    def passenger_count(self):
        return int(np.sum(self.passengers_on_board_od))
    def set_speed(self, mean, std):
        self.speed = float(np.random.normal(mean, std, 1))

    def get_next_stop(self):
        return stops.get_stop_from_name(self.last_stop.next_stop_name)
    def get_distance_to_next_stop(self):
        next_stop = self.get_next_stop()
        return next_stop.get_distance_from_depot() - self.current_distance
    def select_passengers_onboard(self, waiting_passenger_od, maximum):
        selected = []
        tot = 0
        for i in waiting_passenger_od:
            if tot + i < maximum:
                selected.append(i)
                tot += i
            else:
                selected.append( maximum - tot)
                tot += (maximum - tot)
        return np.array(selected).astype(int)
    def __str__(self):
        return f"bus:{self.id} [{self.status}] at {self.last_stop_idx}, total passenger {np.sum(self.passengers_on_board_od)}[{np.sum(self.passengers_on_board_od,axis=0)}]"

    # alighting passenger, all passengers heading to current stop should be off board
    def alight(self):
        stop_idx = self.last_stop.stop_idx
        passengers_off_board = np.array(self.passengers_on_board_od[:,stop_idx])
        self.passengers_on_board_od[:,stop_idx] = 0
        return sum(passengers_off_board)

    def onboard(self, passenger_waiting_od, max_onboard_per_minute=MAX_ONBOARD_PER_MINUTE):
        remaining_capacity = CAPACITY - np.sum(self.passengers_on_board_od)
        max_onboard = min(remaining_capacity, max_onboard_per_minute)
        if (sum(passenger_waiting_od[self.last_stop_idx,:]) <= max_onboard):
            on_passenger = np.array(passenger_waiting_od[self.last_stop_idx,:])
            boarding_completed = True
        else:
            on_passenger = self.select_passengers_onboard(passenger_waiting_od[self.last_stop_idx,:], max_onboard)
            boarding_completed = False
        passenger_waiting_od[self.last_stop_idx, :] -= on_passenger
        self.passengers_on_board_od[self.last_stop_idx,:] += on_passenger
        if self.is_full():
            boarding_completed = True
        return(passenger_waiting_od, sum(on_passenger), boarding_completed)

    def is_full(self):
        return np.sum(self.passengers_on_board_od) >= CAPACITY

    # bus status update for a minute
    def step(self, passenger_waiting_od, action=None):
        on_passenger_count = 0
        off_passenger_count = 0
        stranded_passenger = 0
        self.total_travel_time += 1
        self.passengers_travel_time += np.array(self.passengers_on_board_od)

        next_status = self.status
        if self.status == BusStatus.DECISION:
            if self.is_full():
                minutes = 0
            else:
                minutes = random.randint(0,5)
            self.holding_time = minutes
            if minutes > 0:
                self.status = BusStatus.HOLDING
            else:
                self.status = BusStatus.CRUISING
            print(f"{self}, decision made to hold {minutes} minutes")
        if self.status == BusStatus.CRUISING:
            potential_distance = self.speed / 60
            # not arriving to next stop, continue cruising
            if(self.get_distance_to_next_stop() > potential_distance):
                self.current_distance += potential_distance
                next_status = BusStatus.CRUISING
                print(f"{self}, cruised {potential_distance} km")
            # arriving to next stop
            else:
                self.current_distance += self.get_distance_to_next_stop()
                next_stop_name = self.last_stop.next_stop_name
                self.last_stop = stops.get_stop_from_name(next_stop_name)
                self.last_stop_idx = self.last_stop.stop_idx
                if self.last_stop.is_depot:
                    self.current_distance = 0
                next_status = BusStatus.ARRIVING
                print(f"{self}, cruising to arriving")
        elif self.status == BusStatus.ARRIVING:
            off_passenger_count = self.alight()
            (passenger_waiting_od, on_passenger_count, boarding_completed) = self.onboard(passenger_waiting_od)
            print(f"{self}, {off_passenger_count} alighted, {on_passenger_count} onboarded, boarding_completed: {boarding_completed}")
            if boarding_completed:
                next_status = BusStatus.DECISION
            else:
                next_status = BusStatus.ARRIVING
        elif self.status == BusStatus.HOLDING:
            (passenger_waiting_od, on_passenger_count, boarding_completed) = self.onboard(passenger_waiting_od)
            print(f"{self}, {on_passenger_count} onboarded")
            self.holding_time -= 1
            if(self.holding_time <=0 ):
                next_status = BusStatus.CRUISING
            else:
                next_status = BusStatus.HOLDING
        self.status = next_status
        return passenger_waiting_od, off_passenger_count, on_passenger_count
class BusFleet:
    def __init__(self, n_buses):
        current_package_path = os.path.dirname(__file__)
        csv_file_path = os.path.join(current_package_path, "data", "bus_speed.csv")
        self.speed_df = pd.DataFrame(pd.read_csv(csv_file_path))
        self.buses=[]
        for i in range(N_BUSES):
            bus = Bus(i,status=BusStatus.HOLDING, holding_time=i*20)
            self.buses.append(bus)

    def get_bus_ids(self):
        return [bus.id for bus in self.buses]
    def get_buses_sorted(self):
        sorted_buses = sorted(self.buses, key=lambda x: x.current_distance)
        return sorted_buses

    def get_headways_sorted(self):
        buses = self.get_buses_sorted()
        total_distance = stops.get_route_distance()
        distances = [float(bus.current_distance) for bus in buses]

        distances_a = [(distances[-1] - total_distance)] + distances[:-1]
        distances_b = distances[1:] + [(distances[0] + total_distance)]

        headway_back = [(a-b)%total_distance for a, b in zip(distances,distances_a)]
        headway_ahead = [(a-b)%total_distance for a, b in zip(distances_b,distances)]
        headway_back = [abs(float(item/total_distance)) for item in headway_back] #-0.00000 problem
        headway_ahead = [abs(float(item/total_distance)) for item in headway_ahead]
        return (headway_back, headway_ahead)

    def get_passenger_loads_order_by_id(self):
        result = np.zeros(N_BUSES)
        total_onboard = 0
        for bus in self.buses:
            result[bus.id] = np.sum(bus.passengers_on_board_od)
            total_onboard += np.sum(bus.passengers_on_board_od)
        if total_onboard==0:
            return [0] * N_BUSES
        else:
            return [passenger_cnt / total_onboard for passenger_cnt in result]
    def get_headways_order_by_id(self):
        (headway_back, headway_ahead) = self.get_headways_sorted()
        buses = self.get_buses_sorted()
        headway_back_by_id, headway_ahead_by_id = [0 for i in range(N_BUSES)], [0 for i in range(N_BUSES)]
        for (idx, bus) in enumerate(buses):
            headway_back_by_id[bus.id] = headway_back[idx]
            headway_ahead_by_id[bus.id] = headway_ahead[idx]
        return (headway_back_by_id, headway_ahead_by_id)
    def get_bus_loads(self):
        buses = self.get_buses_sorted()

    def step(self, passenger_waiting_od, actions):
        # total_off_passenger = 0
        # total_on_passenger = 0
        buses_need_decision = []
        onboard_cnt_by_id, offboard_cnt_by_id = np.zeros(N_BUSES), np.zeros(N_BUSES)
        for (bus_position, bus) in enumerate(self.get_buses_sorted()):
            bus_id = bus.id
            action = actions[bus_position]
            passenger_waiting_od, off_passenger_count, on_passenger_count = bus.step(passenger_waiting_od, action)
            onboard_cnt_by_id[bus_id] = on_passenger_count
            offboard_cnt_by_id[bus_id] = off_passenger_count
            # total_off_passenger += off_passenger_count
            # total_on_passenger += on_passenger_count
            if (bus.status == BusStatus.DECISION):
                buses_need_decision.append({'id':bus_id, 'positions':bus_position})
        return (passenger_waiting_od, onboard_cnt_by_id, offboard_cnt_by_id, buses_need_decision)

if __name__ == "__main__":
    fleet = BusFleet(N_BUSES)
    print(fleet.get_buses())
    minute_from, minute_to = 60*6, 60*18
    passenger_waiting_od = np.zeros((N_STOPS,N_STOPS))
    for minute in range(minute_from, minute_to):
        fleet.step(passenger_waiting_od)
        print(minute)
