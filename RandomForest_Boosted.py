import csv 
from pathlib import Path
import random 
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

class RandomForest_Boosted: 
    # We'll store the points in a dictionary of dictionaries where the outer layer of keys stores the point and the inner dictionary contains data specifics
    Luke_Local_directory = Path("CS 4641 Local (Luke)")
    Major_Collector_directory = Path("CS 4641 Major Collector") 
    Principal_Arterial_directory = Path("CS 4641 Principal Arterial")
    Minor_Arterial_directory = Path("CS4641 Minor Arterial")
    Local_directory = Path("Local")

    directories = [Luke_Local_directory, Major_Collector_directory, Principal_Arterial_directory, Minor_Arterial_directory, Local_directory]
    throughput_lanes = {}
    with open("ThroughlanePreprocessing/sheet5_throughlanes.csv", mode = 'r') as file: 
        reader = list(csv.reader(file))
        for i in range(1,len(reader)): 
            point = reader[i][0]
            lanes = reader[i][3]
            throughput_lanes[point] = int(lanes)
    def aggregate(self):
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
        for directory in self.directories:
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
                                day = reader[3][j]
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
                                    if count_sum == 0 : 
                                        counts_proportion = [0 for i in range(15)]
                                    else: 
                                        counts_proportion = [x / count_sum for x in counts_split]
                                else: 
                                    # null readings for all 13 proportions
                                    counts_proportion = [np.nan for i in range(15)]
                                # Adding throughput lanes
                                point = reader[0][1]
                                lanes = self.throughput_lanes[point]
                                row = [road_type_encoded, day_encoded, time, lanes] + counts_proportion
                                

                                # Determing throughput
                                throughput = reader[i][j]
                                if throughput == "": 
                                    throughput = np.nan
                                else: 
                                    throughput = float(throughput)
                                    # Append features and output if output exists
                                    features.append(row)
                                    outputs.append(throughput)
        features = np.array(features)
        outputs = np.array(outputs)
        return features,outputs

    def KFolds(self,features,outputs, k): 
        indexes = list(range(len(features)))
        random.shuffle(indexes)
        splits = np.array_split(indexes, k)
        best_model = None
        best_loss = 0
        best_hyper_parameters = None
        for i in range(k): 
            test_features = features[splits[i]]          
            test_outputs = outputs[splits[i]]   
            remaining_splits = splits[:i] + splits[i+1:]
            train_indices = [idx for split in remaining_splits for idx in split]
            train_features = features[train_indices]   
            train_outputs   = outputs[train_indices]
            model, hyper_parameters = self.train(train_features, train_outputs)
            loss = self.test(model,test_features, test_outputs)
            if loss > best_loss: 
                best_model = model
                best_hyper_parameters = hyper_parameters
                best_loss = loss
        return best_model,best_hyper_parameters,best_loss
    def train(self,features, outputs):
        # Note that the xgboost library works best with numpy arrays so we'll start by converting the input data into a numpy array
        # Create base model
        model = xgb.XGBRegressor(objective = 'reg:squarederror', n_estimators = 100, random_state = 777)
        model.fit(features,outputs)
        # Create parameter grid for second level of k-folds cross validation
        param_grid = {
            'n_estimators': [100, 300],      # Defines the number of sequential boosts (trees)
            'max_depth': [3, 5, 7],          # Defines max depth for each tree
            'learning_rate': [0.01, 0.1],    # Defines learning rate for sequential improvements
            'subsample': [0.8, 1.0],         # Defines fraction of training data (rows) to sample per tree
            'colsample_bytree': [0.8, 1.0],  # Defines the proportion of features to include (randomized)
            'min_child_weight': [1, 5],      # Defines the minimum sum of instance weights (hessian) in a child node
            'gamma': [0, 0.1],               # Defines the minimum loss reduction required to split
            'reg_alpha': [0, 0.1],           # L1 Regularization constant on leaf weights
            'reg_lambda': [1, 1.5]           # L2 Regularization constant on leaf weights
        }
        # Search through combination of hyperparamters
        grid_search = GridSearchCV(estimator=model, param_grid= param_grid, cv = 3, n_jobs = -1, verbose = 1)
        grid_search.fit(features, outputs)
        return grid_search.best_estimator_, grid_search.best_params_
    def test(self,model, features, outputs): 
        prediction = model.predict(features)
        r2 = r2_score(outputs, prediction)
        print(r2)
        return r2
    
random_forest = RandomForest_Boosted()
features, outputs = random_forest.aggregate()
best_model, best_hyper_parameters, best_loss = random_forest.KFolds(features,outputs, 10)
print(f"Best model hyperaparameters: {best_hyper_parameters}, Best model Accuracy {best_loss}" )
         
                