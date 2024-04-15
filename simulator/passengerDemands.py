import numpy as np
import pandas as pd
import importlib.resources
import os
class PassengerDemands:
    def __init__(self):
        current_package_path = os.path.dirname(__file__)
        csv_file_path = os.path.join(current_package_path, "data", "passenger_demands.csv")
        self.passenger_df = pd.DataFrame(pd.read_csv(csv_file_path))
    def get_mean_passenger_per_minute(self, minute, stop_name):
        # self.passenger_df['']
        hour = int(minute/60)
        row = self.passenger_df[(self.passenger_df['hour']==hour) & (self.passenger_df['stop_name']==stop_name)]
        return row['mean_passenger_arriving'].iloc[0]

    def generate_passenger_od(self, as_at_minute, stop_name, stop_idx, total_stops):
        mean_passenger_per_minute = self.get_mean_passenger_per_minute(as_at_minute, stop_name)
        # Generate sample
        passenger_counts = np.random.poisson(mean_passenger_per_minute, 1)[0]

        result = np.zeros(total_stops, dtype=int)
        # distribute demands randomly to downstream stops (including depot)
        if (stop_idx == total_stops - 1):
            result[0] = passenger_counts
        else:
            for _ in range(passenger_counts):
                random_index = np.random.randint(stop_idx+1, total_stops)  # Generate a random index
                result[random_index % total_stops] += 1
        return result

        # print("Number of passengers arriving at the bus stop for every minute:")
        # for minute, count in enumerate(passenger_counts, start=1):
        #     print(f"Minute {minute}: {count} passengers")