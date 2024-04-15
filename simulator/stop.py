import pandas as pd
import os
class BusStop:
    def __init__(self, stop_name, stop_idx, distance, prev_stop_name, next_stop_name, latitude, longitude, is_depot, distance_to_next_stop):
        self.stop_name = stop_name
        self.stop_idx = stop_idx
        self.distance = float(distance)
        self.prev_stop_name = prev_stop_name
        self.next_stop_name = next_stop_name
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.is_depot = is_depot
        self.distance_to_next_stop = distance_to_next_stop

    def get_distance_from_depot(self):
        return self.distance

    def get_distance_to_next_stop(self):
        return self.distance_to_next_stop
    def is_depot(self):
        return self.is_depot

    def __str__(self):
        return f"stop - name: {self.stop_name}, distance: {self.distance}, " \
               f"idx: {self.stop_idx}" \
               f"position: ({self.latitude, self.longitude})"

class BusStops:
    # stop_id, distance_km, latitude, longitude
    def __init__(self):
        current_package_path = os.path.dirname(__file__)
        csv_file_path = os.path.join(current_package_path, "data", "bus_stops.csv")
        stops_df = pd.DataFrame(pd.read_csv(csv_file_path))
        sorted_df = stops_df.sort_values(by='distance_km')
        self.stop_dict = {}
        self.stops = []
        # last stop is the same as first one (depot)
        total_distance = sorted_df.iloc[-1].distance_km
        sorted_stops = pd.DataFrame(sorted_df[:-1])
        total_stops = len(sorted_stops)
        for index, row in sorted_stops.iterrows(): # ignore last stop as it is the depot
            stop_name = str(row['stop_name'])
            distance_km = float(row['distance_km'])
            is_depot = (index == 0)
            latitude, longitude = float(row['latitude']), float(row['longitude'])
            prev_stop_name = sorted_stops.iloc[(index-1) % total_stops]['stop_name']
            next_stop_name = sorted_stops.iloc[(index+1) % total_stops]['stop_name']
            if index == total_stops - 1: # last stop
                distance_to_next_stop = total_distance - distance_km
            else:
                distance_to_next_stop = sorted_stops.iloc[(index+1)]['distance_km'] - distance_km
            if is_depot:
                distance_km = total_distance
            stop = BusStop(stop_name, index, distance_km, prev_stop_name, next_stop_name, latitude,longitude, is_depot, distance_to_next_stop)
            self.stops.append(stop)
            self.stop_dict[stop_name] = stop
    def get_stops_sorted(self):
        return self.stops
    def get_stop_from_idx(self, idx):
        return self.stops[idx]
    def get_stop_from_name(self, name):
        return self.stop_dict[name]

    # def get_next_stop(self, id):
    #     return self.stop_dict[self.stops[id].next_stop_id]
    #
    # def get_prev_stop(self, id):
    #     return self.stop_dict[self.stops[id].prev_stop_id]

    def get_route_distance(self):
        return self.stops[-1].distance

if __name__ == "__main__":
    busStops = BusStops()