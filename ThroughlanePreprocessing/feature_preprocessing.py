import pandas as pd

datapoints_df = pd.read_csv("sheet5_throughlanes.csv")
features_df = pd.read_csv("Road_Inventory_2024/SURFACE_TYPE.csv")

print(datapoints_df.columns)
print(features_df.columns)

merged = datapoints_df.merge(features_df,
                             left_on = "RouteID",
                             right_on = "RouteId",
                             how = "left")
result = merged[
    (merged["AvgMeasure"] >= merged["BeginPoint"]) &
    (merged["AvgMeasure"] <= merged["EndPoint"])
]

result = result[["Point", "SURFACE_TYPEVn"]]

data_merge = datapoints_df.merge(result, on="Point", how="left")

data_merge.to_csv("sheet6_surfaces.csv")