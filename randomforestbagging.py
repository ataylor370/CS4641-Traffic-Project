import csv
from pathlib import Path
import random
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import ParameterGrid


class RandomForest_Bagging:
    Luke_Local_directory = Path("CS 4641 Local (Luke)")
    Major_Collector_directory = Path("CS 4641 Major Collector")
    Principal_Arterial_directory = Path("CS 4641 Principal Arterial")
    Minor_Arterial_directory = Path("CS4641 Minor Arterial")
    Local_directory = Path("Local")

    directories = [
        Luke_Local_directory,
        Major_Collector_directory,
        Principal_Arterial_directory,
        Minor_Arterial_directory,
        Local_directory
    ]

    throughput_lanes = {}
    with open("ThroughlanePreprocessing/sheet5_throughlanes.csv", mode='r') as file:
        reader = list(csv.reader(file))
        for i in range(1, len(reader)):
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
                    with open(file_path, mode='r') as file:
                        reader = list(csv.reader(file))
                        length = len(reader[0])

                        # throughput columns: 6 to 29 (1-indexed) => 5 to 28 (0-indexed)
                        for i in range(5, 29):
                            for j in range(1, length):
                                road_type = reader[2][1].strip()
                                if road_type in road_encodings:
                                    road_type_encoded = road_encodings[road_type]
                                else:
                                    road_type_encoded = -1
                                    print(f"Road type error in point {reader[0][1]}: {reader[2][1]}")

                                day = reader[3][j].strip()
                                if day in day_encodings:
                                    day_encoded = day_encodings[day]
                                else:
                                    continue

                                time = i - 5

                                counts = reader[i + 25][j]
                                if counts != "":
                                    counts_split = [int(x) for x in counts.split(":")]
                                    count_sum = sum(counts_split)
                                    if count_sum == 0:
                                        counts_proportion = [0 for _ in range(15)]
                                    else:
                                        counts_proportion = [x / count_sum for x in counts_split]
                                else:
                                    counts_proportion = [np.nan for _ in range(15)]

                                point = reader[0][1]
                                lanes = self.throughput_lanes[point]

                                row = [road_type_encoded, day_encoded, time, lanes] + counts_proportion

                                throughput = reader[i][j]
                                if throughput == "":
                                    continue
                                else:
                                    throughput = float(throughput)

                                features.append(row)
                                outputs.append(throughput)

        features = np.array(features, dtype=float)
        outputs = np.array(outputs, dtype=float)

        # remove rows with NaNs to keep sklearn trees happy
        valid_rows = ~np.isnan(features).any(axis=1)
        features = features[valid_rows]
        outputs = outputs[valid_rows]

        return features, outputs

    def bootstrap_sample(self, features, outputs):
        n = len(features)
        indices = np.random.choice(n, size=n, replace=True)
        return features[indices], outputs[indices]

    def train(self, features, outputs):
        param_grid = {
            "n_trees": [25, 50, 100],
            "max_depth": [5, 8, 12],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        }

        best_forest = None
        best_params = None
        best_train_score = -float("inf")

        for params in ParameterGrid(param_grid):
            forest = []

            for tree_idx in range(params["n_trees"]):
                X_sample, y_sample = self.bootstrap_sample(features, outputs)

                tree = DecisionTreeRegressor(
                    max_depth=params["max_depth"],
                    min_samples_split=params["min_samples_split"],
                    min_samples_leaf=params["min_samples_leaf"],
                    max_features=params["max_features"],
                    random_state=777 + tree_idx
                )

                tree.fit(X_sample, y_sample)
                forest.append(tree)

            preds = self.predict(forest, features)
            train_score = r2_score(outputs, preds)

            if train_score > best_train_score:
                best_train_score = train_score
                best_forest = forest
                best_params = params

        return best_forest, best_params

    def predict(self, forest, features):
        tree_predictions = np.array([tree.predict(features) for tree in forest])
        return np.mean(tree_predictions, axis=0)

    def test(self, model, features, outputs):
        prediction = self.predict(model, features)
        r2 = r2_score(outputs, prediction)
        rmse = np.sqrt(mean_squared_error(outputs, prediction))
        print(f"R^2: {r2:.4f}, RMSE: {rmse:.4f}")
        return r2

    def KFolds(self, features, outputs, k):
        indexes = list(range(len(features)))
        random.shuffle(indexes)
        splits = np.array_split(indexes, k)

        best_model = None
        best_loss = -float("inf")
        best_hyper_parameters = None

        for i in range(k):
            test_features = features[splits[i]]
            test_outputs = outputs[splits[i]]

            remaining_splits = splits[:i] + splits[i+1:]
            train_indices = [idx for split in remaining_splits for idx in split]
            train_features = features[train_indices]
            train_outputs = outputs[train_indices]

            model, hyper_parameters = self.train(train_features, train_outputs)
            loss = self.test(model, test_features, test_outputs)

            if loss > best_loss:
                best_model = model
                best_hyper_parameters = hyper_parameters
                best_loss = loss

        return best_model, best_hyper_parameters, best_loss


random_forest = RandomForest_Bagging()
features, outputs = random_forest.aggregate()
best_model, best_hyper_parameters, best_loss = random_forest.KFolds(features, outputs, 10)

print(f"Best model hyperparameters: {best_hyper_parameters}")
print(f"Best model R^2: {best_loss:.4f}")