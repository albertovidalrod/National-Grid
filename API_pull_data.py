import datetime
import json
import os
from time import sleep, time

import holidays
import numpy as np
import pandas as pd
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
data_kaggle_dir = os.path.join(current_dir, "data/kaggle")
os.makedirs(data_dir, exist_ok=True)


# Create a dictionary to use the right parameter to pull each year's data
demand_dict = {
    "historic-demand-data-2023": "bf5ab335-9b40-4ea4-b93a-ab4af7bce003",
    "historic-demand-data-2022": "bb44a1b5-75b1-4db2-8491-257f23385006",
    "historic-demand-data-2021": "18c69c42-f20d-46f0-84e9-e279045befc6",
    "historic-demand-data-2020": "33ba6857-2a55-479f-9308-e5c4c53d4381",
    "historic-demand-data-2019": "dd9de980-d724-415a-b344-d8ae11321432",
    "historic-demand-data-2018": "fcb12133-0db0-4f27-a4a5-1669fd9f6d33",
    "historic-demand-data-2017": "2f0f75b8-39c5-46ff-a914-ae38088ed022",
    "historic-demand-data-2016": "3bb75a28-ab44-4a0b-9b1c-9be9715d3c44",
    "historic-demand-data-2015": "cc505e45-65ae-4819-9b90-1fbb06880293",
    "historic-demand-data-2014": "b9005225-49d3-40d1-921c-03ee2d83a2ff",
    "historic-demand-data-2013": "2ff7aaff-8b42-4c1b-b234-9446573a1e27",
    "historic-demand-data-2012": "4bf713a2-ea0c-44d3-a09a-63fc6a634b00",
    "historic-demand-data-2011": "01522076-2691-4140-bfb8-c62284752efd",
    "historic-demand-data-2010": "b3eae4a5-8c3c-4df1-b9de-7db243ac3a09",
    "historic-demand-data-2009": "ed8a37cb-65ac-4581-8dbc-a3130780da3a",
}

# Define url tpo pull data from
url = "https://data.nationalgrideso.com/api/3/action/datastore_search"

# save information from last year
final_year = 2023
if final_year % 4 == 0:
    limit = 48 * 366
else:
    limit = 48 * 365
dict_key = "historic-demand-data-" + str(final_year)
parameters = {"resource_id": demand_dict[dict_key], "limit": limit}

data_request = requests.get(url, params=parameters)
data_request_json = data_request.json()

df_last_year = pd.DataFrame(data_request_json["result"]["records"])
df_last_year.columns = df_last_year.columns.str.lower()
df_last_year.drop(columns=["_id"], axis=1, inplace=True)

save_string = f"historic_demand_year_{final_year}"
df_last_year.to_csv(data_kaggle_dir + f"/{save_string}.csv", index=False)
df_last_year.to_parquet(data_dir + f"/{save_string}.parquet", index=False)

# Create an empty datafram to store the results
df = pd.DataFrame()

# Use a for loop the data for each year and store it in the same dataframe
for i, year in enumerate(range(2009, final_year + 1)):
    if year % 4 == 0:
        limit = 48 * 366
    else:
        limit = 48 * 365
    dict_key = "historic-demand-data-" + str(year)
    parameters = {"resource_id": demand_dict[dict_key], "limit": limit}

    data_request = requests.get(url, params=parameters)
    data_request_json = data_request.json()

    df = pd.concat(
        [df, pd.DataFrame(data_request_json["result"]["records"])],
        axis=0,
        ignore_index=True,
    )

# Change column names to lower case and drop id (row number)
df.columns = df.columns.str.lower()
df.drop(columns=["_id"], axis=1, inplace=True)

# add bank holidays
bank_holiday_england = holidays.UK(
    subdiv="England", years=range(2009, final_year + 2), observed=True
).items()

# Create empty lists to store data
holiday_names = []
holiday_dates = []
holiday_dates_observed = []

for date, name in sorted(bank_holiday_england):
    holiday_dates.append(date)
    holiday_names.append(name)
    # Pop the previous value as observed bank holidays takes place later
    if "Observed" in name:
        holiday_dates_observed.pop()

    holiday_dates_observed.append(np.datetime64(date))

# create column containing holdiays
df["is_holiday"] = df["settlement_date"].apply(
    lambda x: pd.to_datetime(x) in holiday_dates_observed
)
df["is_holiday"] = df["is_holiday"].astype(int)

# Make sure that number-type columns are integer format
int_columns = [column for column in df.columns if column != "period_hour"]
df[int_columns] = df[int_columns].astype(int)

##############################################
# Save file
save_string = f"historic_demand_2009_{final_year}"
df.to_csv(data_kaggle_dir + f"/{save_string}.csv")
df.to_parquet(data_dir + f"/{save_string}.parquet")

# Save metadata
# Current date
current_date = datetime.datetime.now()
current_date_string = current_date.strftime("%d/%m/%Y")
df_metadata = {
    "last updated": current_date_string,
    "last entry date": df["settlement_date"].iloc[-1],
    "columns": {column: str(df[column].dtype) for column in df.columns},
    "dataframe shape": {
        "number rows": df.shape[0],
        "number columns": df.shape[1],
    },
}

# Save the metadata to a JSON file
metadata_path = data_dir + f"/{save_string}_metadata.json"
with open(metadata_path, "w") as file:
    json.dump(df_metadata, file, indent=4)

# Clean the data before saving the dataframe
# sort values
df_clean = df.copy()
df_clean.sort_values(
    by=["settlement_date", "settlement_period"], inplace=True, ignore_index=True
)

# drop columns with nan values
df_clean.drop(columns=["nsl_flow", "eleclink_flow"], axis=1, inplace=True)

# Drop rows where settlement_period value is greater than 48
df_clean.drop(index=df_clean[df_clean["settlement_period"] > 48].index, inplace=True)
df_clean.reset_index(drop=True, inplace=True)

# remove outliers
null_days = df_clean.loc[df_clean["tsd"] == 0.0, "settlement_date"].unique().tolist()

null_days_index = []

for day in null_days:
    null_days_index.append(df_clean[df_clean["settlement_date"] == day].index.tolist())

null_days_index = [item for sublist in null_days_index for item in sublist]

df_clean.drop(index=null_days_index, inplace=True)
df_clean.reset_index(drop=True, inplace=True)

# add column with date
df_clean["period_hour"] = (df_clean["settlement_period"]).apply(
    lambda x: str(datetime.timedelta(hours=(x - 1) * 0.5))
)

df_clean.loc[df_clean["period_hour"] == "1 day, 0:00:00", "period_hour"] = "0:00:00"

# Move the new column
column_to_move = df_clean.pop("period_hour")
df_clean.insert(2, "period_hour", column_to_move)

# Combine hour and date
df_clean["settlement_date"] = pd.to_datetime(
    (df_clean["settlement_date"] + " " + df_clean["period_hour"])
)

# replace index using date
df_clean.set_index("settlement_date", inplace=True)
df_clean.sort_index(inplace=True)

# Make sure that number-type columns are integer format
int_columns = [column for column in df_clean.columns if column != "period_hour"]
df_clean[int_columns] = df_clean[int_columns].astype(int)


########################################
# Save csv
save_string = f"historic_demand_2009_{final_year}_noNaN"
df_clean.to_csv(data_kaggle_dir + f"/{save_string}.csv")
df_clean.to_parquet(data_dir + f"/{save_string}.parquet")

# Save metadata
df_clean_metadata = {
    "last updated": current_date_string,
    "last entry date": str(df_clean.index[-1]),
    "columns": {column: str(df_clean[column].dtype) for column in df_clean.columns},
    "dataframe shape": {
        "number rows": df_clean.shape[0],
        "number columns": df_clean.shape[1],
    },
}

# Save the metadata to a JSON file
metadata_path = data_dir + f"/{save_string}_metadata.json"
with open(metadata_path, "w") as file:
    json.dump(df_clean_metadata, file, indent=4)
