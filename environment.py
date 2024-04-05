import random
import gym
import gym.spaces
from gym.utils import seeding
from enum import Enum
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt


N_FEATURES= 8
N_STOPS = 19
N_BUSES = 6
CAPACITY = 50
REWARD_PER_OFFBOARD = 1
REWARD_PER_ONBOARD = 1
REWARD_PER_HOLDING_TIME = -0.1
REWARD_PER_WAITING_TIME = -0.1

REWARD_PER_PASSENGER_TRAVEL_TIME = -0.01
# REWARD_PER_BUS_TRAVEL_TIME = - 1
REWARD_PER_WAITING_STEP = 0

REWARD_PER_COLLISION = 0
REWARD_PER_BUS_STD = -100
REWARD_PER_STRANDED_PASSENGER = -1
REWARD_PER_BUS_RUNNING_COST = 0
# MAX_TIME = int((21.5 - 5.5) * 60)
MAX_TIME = 3000
# MAX_BUSES = 6
# MIN_BUSES = 3
MAX_HEADWAY = 5
distances_from_start = [0, 170, 675, 1000, 1500, 1800, 2300, 3000, 3900, 4600, 4900, 5400, 5600, 5900, 6300, 6700, 7000,
                        7400, 7500]
travel_times = [3] * N_STOPS

bus_stops = [chr(ord('A') + i) for i in range(N_STOPS)]

def get_travel_ratio_at_stop(stop_idx):
    return (np.sum(travel_times[0:stop_idx]))/np.sum(travel_times)
def getHeadWay():
    # return 12
    return random.randint(8, 13)
    # return random.randint(2, 18)
def clamp_number(number, minimum=0, maximum=10):
    return max(minimum, min(number, maximum))

def distance_to_ahead(lst, target):
    if len(lst) == 0 :
        return 0
    lst_circular = [(idx,x-target) if x > target else (idx,x+1-target) for (idx,x)  in enumerate(lst)]
    min_item = min(lst_circular, key=lambda x: x[1])
    return min_item

def distance_to_behind(lst, target):
    if len(lst) == 0:
        return 0
    lst_circular = [(idx, target-x) if x < target else (idx,target+1-x) for (idx,x) in enumerate(lst)]
    min_item = min(lst_circular, key=lambda x: x[1])
    return min_item

def distance_between_two_buses(ratio1, ratio2):
    dis = abs(ratio1 - ratio2)
    if dis > 0.5:
        return 1.0 - dis
    else:
        return dis
def get_passenger_od(minute):
    probability_distribution = [0.98, 0.01, 0.01]
    # Generate a random number based on the probability distribution
    # random_number = np.random.choice([0, 1], p=probability_distribution,)
    upper_triangular_matrix = np.round(
        np.triu(np.random.choice([0, 1, 2], p=probability_distribution, size=(N_STOPS, N_STOPS)))).astype(int)
    np.fill_diagonal(upper_triangular_matrix, 0)
    return upper_triangular_matrix


class BusStatus(Enum):
    CRUISING = 1
    HOLDING = 2
    ARRIVING = 3


class Bus:
    def __init__(self, id, last_stop=0, status=BusStatus.ARRIVING,
                 passengers_on_board=np.zeros((N_STOPS, N_STOPS)),
                 passengers_travel_time=np.zeros((N_STOPS, N_STOPS))):
        self.id = id
        self.last_stop = last_stop
        self.passengers_on_board_od = passengers_on_board
        self.passengers_travel_time = passengers_travel_time
        self.status = status
        self.holding_time = 0
        self.total_travel_time = 0
        self.time_since_last_stop = 0

    @property
    def passenger_count(self):
        return int(np.sum(self.passengers_on_board_od))

    def get_expected_travel_ratio_future(self, n_minutes, holding=0):
        expected_last_stop = self.last_stop
        if(self.status == BusStatus.HOLDING):
            n_minutes-=self.holding_time
            current_time_since_last_stop = n_minutes if n_minutes>0 else 0
        elif(self.status == BusStatus.ARRIVING):
            n_minutes-= holding
            current_time_since_last_stop = n_minutes if n_minutes>0 else 0
        else:
            current_time_since_last_stop = self.time_since_last_stop + n_minutes
        # expected_minutes_remaining = n_minutes
        second_round = False
        while(current_time_since_last_stop >= travel_times[expected_last_stop]):
            current_time_since_last_stop -= travel_times[expected_last_stop]
            expected_last_stop += 1
            expected_last_stop = expected_last_stop % N_STOPS
            current_time_since_last_stop -= 1
        if (current_time_since_last_stop < 0):
            current_time_since_last_stop = 0
        return (np.sum(travel_times[0:expected_last_stop]) + current_time_since_last_stop) / np.sum(travel_times)

    @property
    def total_trip_ratio(self):
        # currently assuming no traffic issue, travel time is always as planned.
        return (np.sum(travel_times[0:self.last_stop]) + self.time_since_last_stop) /np.sum(travel_times)

    def select_passengers(self, waiting_passenger_od, n):
        selected = []
        tot = 0
        for i in waiting_passenger_od:
            if tot + i < n:
                selected.append(i)
                tot += i
            else:
                selected.append( n - tot)
                tot += (n-tot)
        return np.array(selected).astype(int)

    def step(self, passenger_waiting_od, action=None):
        # new_passenger_od = np.array(passenger_waiting_od)
        on_passenger_count = 0
        off_passenger_count = 0
        stranded_passenger = 0
        self.total_travel_time += 1
        self.passengers_travel_time += np.array(self.passengers_on_board_od)
        # if bus is arriving or holding
        if self.status != BusStatus.CRUISING:
            if self.status == BusStatus.ARRIVING:
                off_passenger = np.array(self.passengers_on_board_od[:, self.last_stop])
                off_passenger_travel_time = self.passengers_travel_time[:, self.last_stop]
                off_passenger_count = np.sum(off_passenger)
                off_passenger_total_travel_time = np.sum(off_passenger * off_passenger_travel_time)
                # off board
                self.passengers_on_board_od[:, self.last_stop] = 0
                self.passengers_travel_time[:, self.last_stop] = 0
                # take holding action
                # print("action:", action)
                holding_time = clamp_number(action[self.last_stop]) if action is not None else 0
                if holding_time > 0:
                    self.status = BusStatus.HOLDING
                    self.holding_time = holding_time
                else:
                    self.status = BusStatus.CRUISING
                    self.time_since_last_stop = 0
                    # holding, passenger on
            elif self.status == BusStatus.HOLDING:
                self.holding_time -= 1
                if self.holding_time <= 0:
                    self.status = BusStatus.CRUISING
                    self.time_since_last_stop = 0
            # currently all get onboard, i.e. no capacity limit
            waiting_passenger_od = np.array(passenger_waiting_od[self.last_stop, :])
            waiting_passengers = np.sum(waiting_passenger_od)
            if waiting_passengers + np.sum(self.passengers_on_board_od) > CAPACITY:
                # print("initial new_passenger_od[self.last_stop]", passenger_waiting_od[self.last_stop])
                on_passenger = self.select_passengers(waiting_passenger_od, CAPACITY-np.sum(self.passengers_on_board_od))
                # print("on_passenger:", on_passenger)
                passenger_waiting_od[self.last_stop] -= on_passenger
                # print("new_passenger_od[self.last_stop]",passenger_waiting_od[self.last_stop])
                # print("np.sum(self.passengers_on_board_od):",np.sum(self.passengers_on_board_od))
            else:
                on_passenger = waiting_passenger_od
                passenger_waiting_od[self.last_stop, :] = 0
            self.passengers_on_board_od[self.last_stop, :] += on_passenger
            on_passenger_count = np.sum(on_passenger)
            stranded_passenger = waiting_passengers - on_passenger_count
        # if bus is moving
        else:
            self.time_since_last_stop += 1
            # arrive next stop
            if self.time_since_last_stop == travel_times[self.last_stop]:
                self.last_stop = (self.last_stop + 1) % N_STOPS
                self.time_since_last_stop = 0
                self.holding_time = 0
                self.status = BusStatus.ARRIVING

        return off_passenger_count, on_passenger_count, np.sum(self.passengers_on_board_od), passenger_waiting_od, stranded_passenger


class State:
    def __init__(self):
        self.n_minute = 0
        self.passenger_waiting_od = get_passenger_od(self.n_minute)
        self.passenger_waiting_times = np.zeros((N_STOPS, N_STOPS))
        self.stranded_passengers = np.zeros(N_STOPS)
        self.next_bus_id = 0
        self.head_way = 0
        self.buses = []
        self.last_bus_left_at_stops = [0]*N_STOPS
        self.add_one_bus()

    def add_one_bus(self):
        self.buses.append(Bus(self.next_bus_id, last_stop=0, status=BusStatus.ARRIVING,
                 passengers_on_board=np.zeros((N_STOPS, N_STOPS)),
                 passengers_travel_time=np.zeros((N_STOPS, N_STOPS))))
        self.next_bus_id += 1
        self.head_way = self.n_minute
        self.next_head_way = getHeadWay()
    def get_buses_in_order(self):
        pos = np.zeros(len(self.buses))
        for (idx, bus) in enumerate(self.buses):
            pos[idx] = bus.total_trip_ratio
        sorted_indices = np.argsort(pos)  # Get indices that would sort x by their position
        # Return the buses sorted by their total_trip_ratio values
        return [self.buses[i] for i in sorted_indices]

    @property
    def shape(self):
        return (N_STOPS, N_FEATURES)
        # return (N_BUSES, 4)
    def reset(self, offset):
        self.n_minute = offset
        self.passenger_waiting_od = get_passenger_od(self.n_minute)
        self.passenger_waiting_times = np.zeros((N_STOPS, N_STOPS))
        self.buses = []
        self.next_bus_id = 0
        self.add_one_bus()
        self.head_way = 0
        self.last_bus_left_at_stops = [0] * N_STOPS
        while(len(self.buses) < N_BUSES):
            reward, (state_stop, state_bus), done, arriving_stops, arriving_buses, rewards, bus_ids = self.step(None)
        return reward, (state_stop, state_bus), done, arriving_stops, arriving_buses, rewards, bus_ids

    def get_single_bus_distribution_reward(self, bus_id=1, n_minutes=2):
        if(len(self.buses) < N_BUSES):
            return 0
        bus_sorted = self.get_buses_in_order()
        bus_sorted_id = [bus.id for bus in bus_sorted]
        idx = bus_sorted_id.index(bus_id)
        #prepend one and append one for easy calculation:
        idx = idx+1
        bus_sorted = [bus_sorted[-1]] + bus_sorted + [bus_sorted[0]]
        bus_sorted_id = [bus_sorted_id[-1]] + bus_sorted_id + [bus_sorted_id[0]]
        bus_sorted_trip_ratio = [bus.total_trip_ratio for bus in bus_sorted]
        bus_sorted_trip_ratio[0] -= 1.0
        bus_sorted_trip_ratio[-1] += 1.0

        ahead_diff = bus_sorted_trip_ratio[idx+1] - bus_sorted_trip_ratio[idx]
        behind_diff = bus_sorted_trip_ratio[idx] - bus_sorted_trip_ratio[idx-1]
        current_diff = 1e4 * (ahead_diff - behind_diff) * (ahead_diff - behind_diff)
        # print("prior ahead_diff:", ahead_diff, "behind diff:", behind_diff, "prior diff:", current_diff)

        bus_ahead = bus_sorted[idx+1]
        bus_ahead_trip_ratio_future = bus_ahead.get_expected_travel_ratio_future(n_minutes)
        bus_behind = bus_sorted[idx-1]
        bus_behind_trip_ratio_future = bus_behind.get_expected_travel_ratio_future(n_minutes)
        bus = bus_sorted[idx]
        bus_trip_ratio_future = bus.get_expected_travel_ratio_future(n_minutes, n_minutes)
        ahead_diff_future = distance_between_two_buses(bus_ahead_trip_ratio_future, bus_trip_ratio_future)
        behind_diff_future = distance_between_two_buses(bus_behind_trip_ratio_future, bus_trip_ratio_future)

        future_diff = 1e4 * (ahead_diff_future - behind_diff_future) * (ahead_diff_future - behind_diff_future)
        return current_diff - future_diff

    def get_bus_distribution(self):
        if(len(self.buses) < N_BUSES):
            return 0
        bus_sorted = [bus.total_trip_ratio for bus in self.get_buses_in_order()]
        bus_sorted = bus_sorted + [1.0+bus_sorted[0]]
        intervals = [bus_sorted[i+1] - bus_sorted[i] for (i,_) in enumerate(bus_sorted) if i<len(bus_sorted)-1]
        deviations = [((item - 1.0 / 6.0) ** 2) for item in intervals]
        return sum(deviations)

    def get_rewards(self):
        if(len(self.buses) < N_BUSES):
            return 0
        bus_sorted = [bus.total_trip_ratio for bus in self.get_buses_in_order()]
        bus_sorted = bus_sorted + [1.0+bus_sorted[0]]
        intervals = [bus_sorted[i+1] - bus_sorted[i] for (i,_) in enumerate(bus_sorted) if i<len(bus_sorted)-1]
        std_dev = [(item - 1/0.6)**2 for item in intervals]
        return std_dev
    def encode(self):
        edges = []
        l_waiting_passengers = np.sum(self.passenger_waiting_od, axis=1)
        l_holding_passengers = np.zeros(N_STOPS)
        l_holding_buses = np.zeros(N_STOPS)
        l_holding_remaining_minutes = np.zeros(N_STOPS)
        l_headway_buses = np.zeros(N_STOPS)
        l_headway_passengers = np.zeros(N_STOPS)
        l_headway_minutes = np.zeros(N_STOPS)
        l_ahead_distance = np.zeros(N_STOPS)
        bus_sorted = self.get_buses_in_order()
        l_travel_ratio = np.zeros(len(bus_sorted))
        l_bus_ahead_distance = np.zeros(N_STOPS)
        l_bus_behind_distance = np.zeros(N_STOPS)

        l_bus_features_trip_ratio = np.zeros(len(self.buses))
        l_bus_features_ahead_distance = np.zeros(len(self.buses))
        l_bus_features_behind_distance = np.zeros(len(self.buses))
        l_bus_features_next_stop = np.zeros(len(self.buses))
        l_bus_features_intervals = np.zeros(len(self.buses))
        l_bus_features_holding = np.zeros(len(self.buses))
        l_bus_feature_arriving = np.zeros(len(self.buses))
        l_bus_feature_passengers = np.zeros(len(self.buses))
        # bus_sorted = bus_sorted + [1.0 + bus_sorted[0]]
        # intervals = [bus_sorted[i + 1] - bus_sorted[i] for (i, _) in enumerate(bus_sorted) if i < len(bus_sorted) - 1]
        # return (max(intervals) - min(intervals)) * np.sum(travel_times)

        for (idx, bus) in enumerate(bus_sorted):
            l_bus_features_trip_ratio[idx] = bus.total_trip_ratio
            l_bus_features_holding[idx] = bus.holding_time
            l_bus_feature_passengers[idx] = bus.passenger_count
            l_bus_features_next_stop[idx] = (bus.last_stop + 1) % N_STOPS
            if bus.status == BusStatus.ARRIVING:
                l_bus_feature_arriving[idx] = 1
            if bus.status == BusStatus.HOLDING or bus.status == BusStatus.ARRIVING:
                l_holding_buses[bus.last_stop] += 1
                l_holding_passengers[bus.last_stop] += np.sum(bus.passengers_on_board_od)
                l_holding_remaining_minutes[bus.last_stop] = max(bus.holding_time,l_holding_remaining_minutes[bus.last_stop]) + 3 if \
                    bus.status == BusStatus.HOLDING else 3
            else:
                l_headway_minutes[bus.last_stop] = min(l_headway_minutes[bus.last_stop], bus.time_since_last_stop)
                l_headway_buses[bus.last_stop] += 1
                l_headway_passengers[bus.last_stop] += np.sum(bus.passengers_on_board_od)
            l_travel_ratio[idx] = bus.total_trip_ratio

        tmp_forward = list(l_bus_features_trip_ratio) + [1.0 + l_bus_features_trip_ratio[0]]
        intervals_forward = [tmp_forward[i + 1] - tmp_forward[i] for (i, _) in enumerate(tmp_forward) if i < len(tmp_forward) - 1]
        l_bus_features_forward_distances = np.array(intervals_forward)
        tmp_backward = [1.0 - l_bus_features_trip_ratio[-1]] + list(l_bus_features_trip_ratio)
        intervals_backward = [tmp_backward[i + 1] - tmp_backward[i] for (i, _) in enumerate(tmp_backward) if i < len(tmp_backward) - 1]
        l_bus_features_backward_distances = np.array(intervals_backward)
        for i in range(N_STOPS):
            stop_travel_ratio = get_travel_ratio_at_stop(i)
            ahead_bus_idx, ahead_bus_trip_ratio = distance_to_ahead(l_travel_ratio, stop_travel_ratio)
            l_bus_ahead_distance[i] = np.sum(travel_times) * ahead_bus_trip_ratio + l_holding_remaining_minutes[i]
            l_bus_ahead_distance[i] = np.sum(travel_times) * ahead_bus_trip_ratio
            behind_bus_idx, behind_bus_trip_ratio = distance_to_behind(l_travel_ratio, stop_travel_ratio)
            l_bus_behind_distance[i] = np.sum(travel_times) * behind_bus_trip_ratio + bus_sorted[behind_bus_idx].holding_time

        for i in range(len(self.buses)):
            behind_bus_idx, behind_bus_trip_ratio = distance_to_behind(l_travel_ratio, l_bus_features_trip_ratio[i])
            l_bus_features_behind_distance[i] = np.sum(travel_times) * behind_bus_trip_ratio + bus_sorted[behind_bus_idx].holding_time
            ahead_bus_idx, ahead_bus_trip_ratio = distance_to_ahead(l_travel_ratio, l_bus_features_trip_ratio[i])
            l_bus_features_ahead_distance[i] = np.sum(travel_times) * ahead_bus_trip_ratio
        # l_headway_times = np.array([self.n_minute]*N_STOPS) - np.array(self.last_bus_left_at_stops)
        l_stop_ids = [i for i in range(0, N_STOPS)]
        l_stop_distance_from_depot = [get_travel_ratio_at_stop(i) for i in range(0, N_STOPS)]
        x = np.concatenate(
            (
                 np.array(l_stop_ids).reshape(-1,1),
                 np.array(l_stop_distance_from_depot).reshape(-1,1),
                 l_waiting_passengers.reshape(-1, 1),
                 # l_holding_buses.reshape(-1, 1),
                 # l_holding_passengers.reshape(-1, 1),
                 # l_holding_remaining_minutes.reshape(-1,1),
                 # l_bus_ahead_distance.reshape(-1,1),
                 # l_bus_behind_distance.reshape(-1,1),
                 self.stranded_passengers.reshape(-1,1)
             ),
            axis=1)
        # print("x.shape", x.shape)
        edge_feature = np.concatenate(
            (l_headway_minutes.reshape(-1,1),
             l_headway_buses.reshape(-1, 1),
             l_headway_passengers.reshape(-1, 1)),
            axis=1)
        # all_feature = np.concatenate((x, edge_feature), axis=1)

        x_tensor = torch.tensor(x, dtype=torch.long)
        l_passenger_waiting_per_bus = self.passengers_waiting_for_buses()
        bus_feature = np.concatenate(
            (
                l_bus_features_trip_ratio.reshape(-1, 1),
                l_bus_features_forward_distances.reshape(-1,1),
                l_bus_features_backward_distances.reshape(-1, 1),
                l_bus_feature_arriving.reshape(-1, 1),
                l_bus_features_holding.reshape(-1,1),
                l_bus_feature_passengers.reshape(-1, 1),
                l_bus_features_next_stop.reshape(-1,1),
                l_passenger_waiting_per_bus.reshape(-1,1)
            ),
            axis=-1
        )

        return (x_tensor, bus_feature)

    def draw(self):
        G = nx.Graph()
        bus_stops = []
        stop_labels = []
        occupied_stops = []
        for bus in self.buses:
            if (bus.time_since_last_stop == 0):
                occupied_stops.append(bus.last_stop)
        for i in range(N_STOPS):
            stop_id = chr(ord('A') + i)
            waiting_passengers = int(np.sum(self.passenger_waiting_od[i]))
            bus_stops.append(f"{stop_id}")
            stop_labels.append(f"{waiting_passengers}")
        edge_distances = [3] * N_STOPS
        # Add bus stops as nodes with their positions
        G.add_nodes_from(bus_stops)
        edges = []
        for i in range(N_STOPS - 1):
            edges.append((bus_stops[i], bus_stops[i + 1], {"distance": edge_distances[i]}))
        edges.append((bus_stops[N_STOPS - 1], bus_stops[0], {"distance": edge_distances[N_STOPS - 1]}))

        # Add edges between bus stops with distances
        G.add_edges_from(edges)

        # Automatically generate circular layout for the bus stops
        pos = nx.circular_layout(G, scale=6, center=(0, 0))
        label_pos = nx.circular_layout(G, scale=7, center=(0, 0))
        # G.add_nodes_from(stop_labels)

        # Update the positions in the node attributes
        for stop, position in pos.items():
            G.nodes[stop]["pos"] = position
        for stop, position in label_pos.items():
            stop_idx = ord(stop) - ord('A')
            stop_label_node = f"({stop}:{stop_labels[stop_idx]})"
            G.add_node(stop_label_node, pos=position, node_type="label")

        # Create bus position nodes for each bus and place them on the circular route
        for bus in self.buses:
            current_stop = bus_stops[bus.last_stop]
            next_stop = bus_stops[(bus.last_stop + 1) % N_STOPS]
            progress = bus.time_since_last_stop / travel_times[bus.last_stop]
            edge_distance = G[current_stop][next_stop]["distance"]

            # Calculate the bus position along the edge using linear interpolation
            pos_a = G.nodes[current_stop]["pos"]
            pos_b = G.nodes[next_stop]["pos"]
            # bus_x = pos_a[0] + (progress * (pos_b[0] - pos_a[0]) / edge_distance)
            # bus_y = pos_a[1] + (progress * (pos_b[1] - pos_a[1]) / edge_distance)
            bus_x = pos_a[0] + (progress * (pos_b[0] - pos_a[0]))
            bus_y = pos_a[1] + (progress * (pos_b[1] - pos_a[1]))

            # Add the bus position node as a smaller node with a different color
            bus_position_node = f"{bus.id}({bus.passenger_count})"
            if (bus.status != BusStatus.HOLDING):
                G.add_node(bus_position_node, pos=(bus_x, bus_y), node_type="bus")
            else:
                bus_position_node = f"{bus_position_node}-{int(bus.holding_time)}"
                G.add_node(bus_position_node, pos=(bus_x, bus_y), node_type="holding")

        # Separate the nodes into bus stops and buses
        bus_nodes = [node for node, data in G.nodes(data=True) if data.get("node_type") == "bus"]
        holding_nodes = [node for node, data in G.nodes(data=True) if data.get("node_type") == "holding"]
        stop_nodes = [node for node, data in G.nodes(data=True) if data.get("node_type") is None]

        # Visualize the graph with bus positions and different node attributes
        pos = nx.get_node_attributes(G, "pos")
        labels = {node: node for node in G.nodes()}

        # Node colors: Red for buses, Blue for bus stops
        node_colors = ["r" if node in bus_nodes else "g" if node in holding_nodes else "b" for node in G.nodes()]

        # Node sizes: Smaller for buses, Larger for bus stops
        node_sizes = [800 if node in bus_nodes else 200 if node in stop_nodes else 1200 for node in G.nodes()]

        plt.text(0.5, -0.1, f"Time: {self.n_minute}", ha='center', va='center', transform=plt.gca().transAxes)
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=node_sizes, node_color=node_colors, font_size=6,
                font_color='white')
        plt.pause(1)  # Pause for 1 second to refresh the plot
        plt.clf()
        plt.show(block=False)  # Use block=False to update the existing plot
    def passengers_waiting_for_buses(self):
        bus_sorted = self.get_buses_in_order()
        bus_next_stops = np.zeros(len(self.buses))
        for idx, bus in enumerate(bus_sorted):
            bus_next_stops[idx] = bus.last_stop + (1 if bus.status == BusStatus.CRUISING else 0)
            # arriving_stops = [bus.last_stop for bus in self.buses if bus.status == BusStatus.ARRIVING]
        passenger_waiting_per_stop = np.sum(self.passenger_waiting_od, axis=1)
        passenger_waiting_per_bus = np.zeros(len(self.buses))
        for idx, stop_from in enumerate(bus_next_stops):
            next_idx = (idx + 1) % len(bus_sorted)
            stop_to = bus_next_stops[next_idx]
            if stop_from < stop_to:
                passenger_waiting_per_bus[idx] = np.sum(passenger_waiting_per_stop[int(stop_from):int(stop_to)])
            elif stop_from == stop_to:
                passenger_waiting_per_bus[idx] = np.sum(passenger_waiting_per_stop)
            else:
                passenger_waiting_per_bus[idx] = np.sum(passenger_waiting_per_stop[int(stop_from):]) + \
                                                 np.sum(passenger_waiting_per_stop[:int(stop_to)])
        # adding tailing 0s to make it fixed size
        return passenger_waiting_per_bus
    def step(self, action):
        self.n_minute += 1
        new_passenger_waiting_od = get_passenger_od(self.n_minute)
        self.passenger_waiting_od += new_passenger_waiting_od
        total_off_passenger_count = 0
        total_on_passenger_count = 0
        total_passenger_travel_time = 0
        total_passenger_holding_time = 0
        total_stranded_passenger = 0
        total_passenger_waiting_time = np.sum(new_passenger_waiting_od)
        passenger_waiting_time_for_buses = np.zeros(N_BUSES)
        if len(self.buses) < N_BUSES:
            action = None
        if self.n_minute == (self.head_way + self.next_head_way) and len(self.buses) < N_BUSES:
            self.add_one_bus()

        stops = []
        self.stranded_passengers = np.zeros(N_STOPS)
        crusing_buses = 0
        rewards = np.zeros(N_BUSES)
        bus_sorted = self.get_buses_in_order()
        for idx, bus in enumerate(bus_sorted):
            bus_id = bus.id
            prev_status = bus.status
            if (bus.status == BusStatus.ARRIVING) and action is not None:
                holding = True
            else:
                holding = False
            if (bus.status == BusStatus.CRUISING):
                crusing_buses += 1
            off_passenger_count, on_passenger_count, total_onboard_count, self.passenger_waiting_od, stranded_passengers = bus.step(
                    self.passenger_waiting_od, action)

            if holding:
                holding_minutes = action[bus.last_stop]
                # rewards[bus.last_stop] = holding_minutes * - 10 \
                #                         + self.get_single_bus_distribution_reward(bus.id, holding_minutes)
                rewards[idx] = off_passenger_count * REWARD_PER_OFFBOARD + \
                                on_passenger_count * REWARD_PER_ONBOARD +\
                                holding_minutes * REWARD_PER_HOLDING_TIME

            self.stranded_passengers[bus.last_stop] += stranded_passengers
            total_stranded_passenger += stranded_passengers
            if holding:
                action[bus.last_stop] = 0 #make sure the other buses at the same stop doesn't hold
                total_passenger_holding_time += bus.passenger_count
            if(prev_status != BusStatus.CRUISING and bus.status == BusStatus.CRUISING):
                self.last_bus_left_at_stops[bus.last_stop] = self.n_minute
            total_off_passenger_count += off_passenger_count
            total_on_passenger_count += on_passenger_count
            total_passenger_travel_time += total_onboard_count
            stops.append(bus.last_stop)
        reward = self.get_bus_distribution() * REWARD_PER_BUS_STD
        done = self.n_minute > MAX_TIME

        arriving_stops = []
        arriving_buses = []
        bus_ids = []
        bus_next_stops = np.zeros(len(self.buses))
        for idx, bus in enumerate(bus_sorted):
            if bus.status == BusStatus.ARRIVING:
                arriving_buses.append(idx)
                arriving_stops.append(bus.last_stop)
                bus_ids.append(bus.id)
            bus_next_stops[idx] = bus.last_stop + (1 if bus.status == BusStatus.CRUISING else 0)
            # arriving_stops = [bus.last_stop for bus in self.buses if bus.status == BusStatus.ARRIVING]
        passenger_waiting_per_bus = self.passengers_waiting_for_buses()
        passenger_waiting_per_bus = np.array(list(passenger_waiting_per_bus) + [0]*(N_BUSES-len(self.buses)))
        rewards += passenger_waiting_per_bus * REWARD_PER_WAITING_TIME
        passenger_traval_time_per_bus = [np.sum(bus.passengers_travel_time) for bus in bus_sorted]
        passenger_traval_time_per_bus = np.array(list(passenger_traval_time_per_bus) + [0]*(N_BUSES-len(self.buses)))
        rewards += passenger_traval_time_per_bus * REWARD_PER_PASSENGER_TRAVEL_TIME

        # arriving_buses = [bus.id-1 for bus in self.buses if bus.status == BusStatus.ARRIVING]
        print("reward:", reward, "total buses:",  len(self.buses))
        return reward, self.encode(), done, arriving_stops, arriving_buses, rewards, bus_ids

class BusEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # Register the enviro nment
    def __init__(self):
        super(BusEnv, self).__init__()
        self.minutes = 0
        self._state = State()
        self.action_space = gym.spaces.Box(low=0, high=5, shape=(N_STOPS,))
        self._max_episode_steps = 240
        # self.action_space = gym.spaces.Box(low=0, high=5)
        self.observation_space = gym.spaces.Box(
            low=0, high=5,
            shape=self._state.shape, dtype=np.float32)

        # self.seed()

    def reset(self, pre_steps=0, random=False):
        """
        reset state to fresh start. i.e, no running bus, no awaiting passengers.
        we choose route randomly, and choose current time offset to zero or randomly,
        and we backward current offset further by pre_steps steps. pre_steps is adjusted if reached the minimum time range.

        :param pre_steps: the number of minutes we move backward for staging
        :param random: bool. we set offset to route start time if false, otherwise set offset randomly between start and end time.
        :return observation
        """
        self.minutes = 0
        return self._state.reset(pre_steps)

    @property
    def buses(self):
        return len(self._state.buses)

    @property
    def bus_stops(self):
        stops = []
        for bus in self._state.buses:
            stops.append(bus.last_stop)
        return sorted(stops)
    def step_old(self, action, render=False):
        total_reward = 0
        while True:
            reward, (obs, bus_feature), done, arriving_stops, arriving_buses, rewards, bus_ids = self._state.step(action)
            total_reward += reward
            if render:
                self.render()
            # status = self.state.get_trip_status(running=False)
            # print("obs.shape:", obs.shape)
            if len(arriving_stops) > 0 or done:
                return obs, total_reward, done, arriving_stops, arriving_buses, rewards, bus_ids
    def step(self, action, render=False):
        reward, (stop_feature, bus_feature), done, arriving_stops, arriving_buses, rewards, bus_ids = self._state.step(action)
        if render:
            self.render()
        return (stop_feature, bus_feature), reward, done, arriving_stops, arriving_buses, rewards, bus_ids

    def get_time(self):
        return self.state._offset

    def render(self, mode='human', close=False):
        self._state.draw()

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]


def choose_action(x_stops, x_buses):
    return random.randint(0, MAX_HEADWAY)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    gym.register(id='BusEnv-v0', entry_point='your_module:BusEnv')
    env = BusEnv()
    while (env.buses < N_BUSES):
        (state, state_bus), total_reward, done, arriving_stops, arriving_buses, rewards, _ = env.step(None)

    action = None
    while (True):
        state, reward, done, arriving_stops, arriving_buses, _, _ = env.step(action, True)
        action = [0] * N_STOPS
        if len(arriving_buses)>0:
            for idx, stop_idx in enumerate(arriving_stops):
                bus_idx = arriving_buses[idx]
                state_stops_rotated = np.concatenate((state[0][stop_idx:], state[0][0:stop_idx]))
                state_buses_rotated = np.concatenate((state[1][bus_idx:], state[1][0:bus_idx]))
                bus_action = choose_action(state_stops_rotated, state_buses_rotated)
                action[arriving_stops[idx]] = int(bus_action)
        else:
            action = None


