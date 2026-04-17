import pandas as pd
from pathlib import Path
import csv
#split points into train and test

#fetch data points and put into csv file
def parse_file(file):
  header = {}
  hour_rows = []

  with open(file) as f:
    reader = csv.reader(f)
    for row in reader:
      if not any(field.strip() for field in row):
        break
      key = row[0].strip()
      match key:
        case "Point:":
          header.update({"point": row[1].strip()})
        case "Location:":
          header.update({"location": row[1].strip()})
        case "Road-Type:":
          header.update({"road_type": row[1].strip()})
        case "Day:":
          header.update({"days_of_week": [d.strip() for d in row[1:] if d.strip()]})
        case "Date:":
          header.update({"dates": [d.strip() for d in row[1:] if d.strip()]})
        case _:
          if key[0].isdigit() and ":" in key:
            hour = key.split(":")[0]
            values = [v.strip() if v.strip() else None for v in row[1:len(header["days_of_week"]) + 1]]
            hour_rows.append([hour] + values)
  df = pd.DataFrame(hour_rows, columns=["time"] + header["days_of_week"])
  df = df.melt(id_vars="time", var_name="days_of_week", value_name="hourly_throughput")

  day_to_date = dict(zip(header["days_of_week"], header["dates"]))
  df["dates"] = df["days_of_week"].map(day_to_date)

  df["point"] = header["point"]
  df["road_type"] = header["road_type"]

  df = df.dropna(subset=["hourly_throughput"])
  df["hourly_throughput"] = df["hourly_throughput"].astype(int)

  return df[["point", "road_type", "days_of_week", "dates", "time", "hourly_throughput"]]

local_luke = [parse_file(p) for p in (Path(__file__).parent.parent / "CS 4641 Local (Luke)").glob("*.csv")]
local = [parse_file(p) for p in (Path(__file__).parent.parent / "Local").glob("*.csv")]
prin_art = [parse_file(p) for p in (Path(__file__).parent.parent / "CS 4641 Principal Arterial").glob("*.csv")]
maj_coll = [parse_file(p) for p in (Path(__file__).parent.parent / "CS 4641 Major Collector").glob("*.csv")]
min_art = [parse_file(p) for p in (Path(__file__).parent.parent / "CS4641 Minor Arterial").glob("*.csv")]
all_dfs = [*local_luke, *local, *prin_art, *maj_coll, *min_art]
result = pd.concat(all_dfs, ignore_index=True)
lanes_df = pd.read_csv("throughlanes_data.csv")
result = result.merge(lanes_df[["Point", "THROUGH_LANESVn"]], left_on="point", right_on="Point", how="left")
result = result.drop(columns="Point")
result = result.rename(columns={"THROUGH_LANESVn": "through_lanes"})
result.to_csv("data_points.csv", index=False)