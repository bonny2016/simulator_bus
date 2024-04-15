from simulator.stop import BusStops
from simulator.globals import N_STOPS, N_BUSES, MINUTE_FROM, MINUTE_TO, CAPACITY, N_BUS_FEATURES, N_STOP_FEATURES, N_ACTIONS
import numpy as np
from simulator import passengerDemands as passenger
import gym
import gym.spaces
from simulator.memory import Memory
from simulator.bus import BusStatus, BusFleet
import statistics

class BusEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(BusEnv, self).__init__()
        self.reset()

    def reset(self, pre_steps=0, random=False):
        self.stops = BusStops()
        self.fleet = BusFleet(N_BUSES)
        self.demands = passenger.PassengerDemands()
        self._stop_arr = self.stops.get_stops_sorted()
        self.route_distance = self.stops.get_route_distance()
        self.passenger_waiting_od = np.zeros((N_STOPS,N_STOPS))
        self.current_minutes = MINUTE_FROM
        self.action_space = gym.spaces.Box(low=0, high=5, shape=(N_STOPS,))
        self.max_episode_steps = MINUTE_TO - MINUTE_FROM
        self.bus_ids = list(self.fleet.get_bus_ids())
        self.memory = Memory(self.bus_ids)
        self.current_bus_states = None
        self.current_bus_indices_needs_decision = []
        # ((new_bus_states, new_stop_states), total_rewards, self.is_done, buses_need_decision)
        return (([],[]), 0, False, [])
    def get_new_demand_od(self):
        n_minute = self.current_minutes
        n_stops = len(self._stop_arr)
        od_matrix = np.zeros([n_stops, n_stops])
        for (idx, stop) in enumerate(self._stop_arr):
            stop_name = stop.stop_name
            od_matrix[idx, :] = self.demands.generate_passenger_od(n_minute,
                                                              stop_name=stop_name,
                                                              stop_idx=idx,
                                                              total_stops=n_stops)
        return od_matrix
    def get_fleet_state(self):
        # (h-,h+,stop_idx,load,remaining_holding_minutes)
        fleet_state_matrix = np.zeros((N_BUSES, N_BUS_FEATURES))
        for idx, bus in enumerate(self.fleet.get_buses_sorted()):
            fleet_state_matrix[idx,:] = [float(bus.current_distance/self.route_distance), bus.last_stop_idx, np.sum(bus.passengers_on_board_od)/CAPACITY, bus.holding_time]
        return fleet_state_matrix

    def get_bus_states(self):
        fleet_states = self.get_fleet_state()
        bus_idx_needs_decision = self.bus_indices_needs_decision()
        result = []
        for idx in bus_idx_needs_decision:
            bus_state = np.concatenate((fleet_states[idx:], fleet_states[0:idx]))
            result.append(bus_state)
        return result

    def get_stop_state(self):
        stops = self._stop_arr
        result = np.zeros((N_STOPS, N_STOP_FEATURES))
        for i in range(N_STOPS):
            result[i,0] = np.sum(self.passenger_waiting_od[i,:])
            result[i,1] = stops[i].get_distance_from_depot()/self.route_distance
        return result

    # TODO: add passenger load
    def get_rewards_order_by_id(self, onboard_cnt_by_id, offboard_cnt_by_id, action_order_by_id):
        passengers_cnts = onboard_cnt_by_id + offboard_cnt_by_id
        (headway_back_by_id, headway_ahead_by_id) = self.fleet.get_headways_order_by_id()
        mean_headway = 1.0/N_BUSES
        is_any_negative = any(x < 0 for x in headway_back_by_id) or any(x < 0 for x in headway_ahead_by_id)
        if (is_any_negative):
            print("here")

        headway_var = [np.var([h_back - mean_headway, h_ahead - mean_headway]) for (h_back, h_ahead) in zip(headway_back_by_id, headway_ahead_by_id)]
        is_any_nan = np.isnan(headway_var).any()
        if is_any_nan:
            print("here")
        passenger_load = self.fleet.get_passenger_loads_order_by_id()
        is_any_nan = np.isnan(passenger_load).any()
        if is_any_nan:
            print("here")
        passenger_mean = np.mean(passenger_load)
        passenger_load_var = [(load - passenger_mean)**2  for load in passenger_load]
        result =  [ float(-load - headway - hold/N_ACTIONS) for headway, load, hold in zip(headway_var, passenger_load_var, action_order_by_id)]
        return result
    def bus_indices_needs_decision(self):
        buses = self.fleet.get_buses_sorted()
        bus_indices_needs_action = []
        for idx, bus in enumerate(buses):
            if (bus.status == BusStatus.DECISION):
                bus_indices_needs_action.append(idx)
        return bus_indices_needs_action

    def step(self, actions):
        # s_, r, done, _ = env.step(a)
        new_demand_od = self.get_new_demand_od()
        self.passenger_waiting_od += new_demand_od
        buses = self.fleet.get_buses_sorted()
        real_actions = [None] * N_BUSES
        # current_states = self.current_bus_states
        # current_bus_indices_needs_decision = self.current_bus_indices_needs_decision
        action_order_by_id = np.zeros(N_BUSES)
        for idx in self.current_bus_indices_needs_decision:
            real_actions[idx] = actions[idx]
            bus_id = buses[idx].id
            action_order_by_id[bus_id] = actions[idx]
            self.memory.remember_temp_action(bus_id, actions[idx])
        #buses_need_decision:[{'id':bus_id, 'positions':bus_position}]
        (self.passenger_waiting_od, onboard_cnt_by_id, offboard_cnt_by_id, buses_need_decision) = \
            self.fleet.step(self.passenger_waiting_od, real_actions)
        rewards_by_bus_id = self.get_rewards_order_by_id(onboard_cnt_by_id, offboard_cnt_by_id, action_order_by_id)
        for (bus_id, reward) in enumerate(rewards_by_bus_id):
            self.memory.remember_temp_reward(bus_id, reward)
        total_rewards = np.sum(rewards_by_bus_id)

        self.current_bus_indices_needs_decision = self.bus_indices_needs_decision()
        if(len(self.current_bus_indices_needs_decision) == 0):
            new_bus_states = []
            new_stop_state = []
        else:
            buses = self.fleet.get_buses_sorted()
            new_bus_states = self.get_bus_states()
            new_stop_state = self.get_stop_state()
            for (idx, bus_position) in enumerate(self.current_bus_indices_needs_decision):
                bus_id = buses[bus_position].id
                bus_state = (new_bus_states[idx], new_stop_state)
                self.memory.remember_temp_result(bus_id, bus_state, self.is_done)
        self.current_minutes += 1
        s_next = [(s_bus, new_stop_state) for s_bus in new_bus_states]
        return(s_next, total_rewards, self.is_done, buses_need_decision)

    def get_current_data(self):
        return self.memory.pop_all()
    @property
    def is_done(self):
        return self.current_minutes == self.max_episode_steps

if __name__ == "__main__":
    env = BusEnv()
    while(True):
        a = [None] * N_BUSES
        print(env.current_minutes)
        s_, r, done, need_decision = env.step(a)
        if done:
            break