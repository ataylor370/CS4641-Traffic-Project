import csv 
from pathlib import Path
import random 
from itertools import batched

class RandomForest_Boosted: 
    # We'll store the points in a dictionary of dictionaries where the outer layer of keys stores the point and the inner dictionary contains data specifics
    Luke_Local_directory = Path("CS 4641 Local (Luke)")
    Major_Collector_directory = Path("CS 4641 Major Collector") 
    Principal_Arterial_directory = Path("CS 4641 Principal Arterial")
    Minor_Arterial_directory = Path("CS4641 Minor Arterial")
    Local_directory = Path("Local")

    directories = [Luke_Local_directory, Major_Collector_directory, Principal_Arterial_directory, Minor_Arterial_directory, Local_directory]
    """ 
    Initial attempt 
    def aggregate(self): 
    # Will be used later as a way to randomly select points
        points = []
        data = {}
        for directory in self.directories: 
            for file_path in directory.iterdir():
                if file_path.is_file():    
                    with open(file_path, mode = 'r') as file: 
                        reader = list(csv.reader(file)) 
                        point = reader[0][1]
                        points.append(point)
                        length = len(reader[0])
                        data[point] = {
                            "location": reader[1][1],
                            "Road-type": reader[2][1],
                            "Points" : [{
                                "day": reader[3][i],
                                "date": reader[4][i],
                                "12AM V/C": (reader[5][i],reader[30][i]) ,
                                "1AM V/C": (reader[6][i], reader[31][i]),
                                "2AM V/C": (reader[7][i],reader[32][i]),
                                "3AM V/C": (reader[8][i],reader[33][i]),
                                "4AM V/C": (reader[9][i],reader[34][i]),
                                "5AM V/C": (reader[10][i],reader[35][i]),
                                "6AM V/C": (reader[11][i],reader[36][i]),
                                "7AM V/C": (reader[12][i],reader[37][i]),
                                "8AM V/C": (reader[13][i],reader[38][i]),
                                "9AM V/C": (reader[14][i],reader[39][i]),
                                "10AM V/C": (reader[15][i],reader[40][i]),
                                "11AM V/C": (reader[16][i],reader[41][i]),
                                "12PM V/C": (reader[17][i],reader[42][i]),
                                "1PM V/C": (reader[18][i],reader[43][i]),
                                "2PM V/C": (reader[19][i],reader[44][i]),
                                "3PM V/C": (reader[20][i],reader[45][i]),
                                "4PM V/C": (reader[21][i],reader[46][i]),
                                "5PM V/C": (reader[22][i],reader[47][i]),
                                "6PM V/C": (reader[23][i],reader[48][i]),
                                "7PM V/C": (reader[24][i],reader[49][i]),
                                "8PM V/C": (reader[25][i],reader[50][i]),
                                "9PM V/C": (reader[26][i],reader[51][i]),
                                "10PM V/C": (reader[27][i],reader[52][i]),
                                "11PM V/C": (reader[28][i],reader[53][i])
                            } for i in range(1,length)]
                        }
        return data 
    """
    def aggregate(self):
        # The features we are recording 
        data = []
        outputs = []
        for directory in self.directories: 
            for file_path in directory.iterdir():
                if file_path.is_file():    
                    with open(file_path, mode = 'r') as file: 
                        reader = list(csv.reader(file))
                        # Collect features: 
                        # Road-type (Principal arterial, Minor arterial, Major Collector, Local to be encoded 1,2,3,4 respectively)
                        # Day (Mon - Fri to be encoded 1-5 respectively)
                        # Time (12AM - 11PM to be encoded 0 - 23 respectively)
                        # Motorcycle Proportion 
                        # Passanger Car Proportion
                        # Pickups, panels, vans Proportion
                        # Buses Proportion
                        # Single-unit trucks (2 axle) Proportion
                        # Single-unit trucks (3 axle) Proportion
                        # Single-unit trucks (4+ axle) Proportion
                        # Single-trailer trucks (3/4 axle) Proportion
                        # Single-trailer trucks (5 axle) Proportion
                        # Single-trailer trucks (6+ axle) Proportion
                        # Multi-trailer trucks (5 or less axle) Proportion
                        # Multi-trailer trucks (6 axle) Proportion
                        # Multi-trailer trucks (7 or more axle) Proportion
                        # Note proportions will be between 0 and 1 
                        
                        # Collect output

    def KFolds(self,data, points, k): 
        shuffled_points = points.copy()
        random.shuffle(shuffled_points)
        splits = list(batched(shuffled_points, len(shuffled_points) // k))
        best_model = None
        best_loss = float('inf')
        for i in range(k): 
            test_points = (data[j] for j in splits[i])
            remaining_splits = [splits[:i] + splits[i+1:]]
            train_indices = [idx for split in remaining_splits for idx in split]
            train_points = (data[idx] for idx in train_indices)
            model = self.train(train_points)
            loss = self.test(model,test_points)
            if loss < best_loss: 
                best_model = model
                best_loss = loss
        return best_model,loss
    def train(self,data):
        # Note that the xgboost library works best with numpy arrays so we'll start by converting the input data into a numpy array
        formatted_data = []
        return model
    def test(self,model, data): 
        return loss
    
Luke_Local_directory = Path("CS 4641 Local (Luke)")
Major_Collector_directory = Path("CS 4641 Major Collector") 
Principal_Arterial_directory = Path("CS 4641 Principal Arterial")
Minor_Arterial_directory = Path("CS4641 Minor Arterial")
Local_directory = Path("Local")

directories = [Luke_Local_directory, Major_Collector_directory, Principal_Arterial_directory, Minor_Arterial_directory, Local_directory]
features = []
outputs = []
road_encodings = {
    "Principal Arterial": 1,
    "Minor Arterial": 2,
    "Major Collector": 3,
    "Local": 4
}
day_encodings = {
    "Mon": 1,
    "Tue": 2, 
    "Wed": 3,
    "Thu": 4, 
    "Fri": 5 
}
for directory in directories: 
    for file_path in directory.iterdir():
        if file_path.is_file():    
            with open(file_path, mode = 'r') as file: 
                reader = list(csv.reader(file))
                length = len(reader[0])
                # Cover All time stamps
                # Time stamps for volume start at column 6 and end at 29 (one indexed)
                for i in range(5,29): 
                    # Cover All entries
                    for j in range(1,length):
                        # determining road-type
                        road_type = reader[2][1]
                        road_type = road_type.strip()
                        if road_type in road_encodings: 
                            road_type_encoded = road_encodings[road_type]
                        else: 
                            road_type_encoded =-1
                            print(f"road type error in point {reader[0][1]} with the following road-type: {reader[2][1]}")
                        
                        # Determining day
                        day = reader[3][i]
                        day = day.strip()
                        if day in day_encodings: 
                            day_encoded = day_encodings[day]
                        
                        # Determining time 
                        time = i - 5
                        # Determining proportions 
                        # Time stamps for classifications start at column 31 and end at 54 (one indexed)
                        # Note counts will be in the following format: 4:44:1:0:0:0:0:0:0:0:0:0:0:0:0
                        counts = reader[i+25][j]
                        if counts != "": 
                            counts_split = [int(x) for x in counts.split(":")]
                            count_sum = sum(counts_split)
                            counts_proportion = [x / count_sum for x in counts_split]
                        else: 
                            # null readings for all 13 proportions
                            counts_proportion = [np.nan for i in range(15)]
                        row = [road_type_encoded, day_encoded, time] + counts_proportion
                        # Determing throughput
                        throughput = reader[i][j] 
                        if throughput == "": 
                            throughput = np.nan
                        # Append features and output if output exists
                        else: 
                            features.append(row)
                            outputs.append(throughput)
                        
features = np.array(features)
outputs = np.array(outputs)
                        
                