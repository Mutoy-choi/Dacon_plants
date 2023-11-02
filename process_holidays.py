import pandas as pd

# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Combine all fixed and lunar holidays and format them correctly
all_holidays = [
    # Fixed holidays from 2019 to 2023
    '2019-01-01', '2019-03-01', '2019-05-05', '2019-06-06', '2019-08-15', '2019-10-03', '2019-10-09', '2019-12-25',
    '2020-01-01', '2020-03-01', '2020-05-05', '2020-06-06', '2020-08-15', '2020-10-03', '2020-10-09', '2020-12-25',
    '2021-01-01', '2021-03-01', '2021-05-05', '2021-06-06', '2021-08-15', '2021-10-03', '2021-10-09', '2021-12-25',
    '2022-01-01', '2022-03-01', '2022-05-05', '2022-06-06', '2022-08-15', '2022-10-03', '2022-10-09', '2022-12-25',
    '2023-01-01', '2023-03-01', '2023-05-05', '2023-06-06', '2023-08-15', '2023-10-03', '2023-10-09', '2023-12-25',
    # Lunar holidays from 2019 to 2023
    '2019-02-04', '2019-02-05', '2019-02-06',
    '2020-01-24', '2020-01-25', '2020-01-26',
    '2021-02-11', '2021-02-12', '2021-02-13',
    '2022-02-01', '2022-02-02', '2022-02-03',
    '2023-01-22', '2023-01-23', '2023-01-24',
    # Additional lunar holidays
    '2019-09-12', '2019-09-13', '2019-09-14',
    '2020-09-30', '2020-10-01', '2020-10-02',
    '2021-09-20', '2021-09-21', '2021-09-22',
    '2022-09-09', '2022-09-10', '2022-09-11',
    '2023-09-28', '2023-09-29', '2023-09-30'
]

# Function to process each dataset
def process_dataset(dataframe):
    # Convert the 'timestamp' column to datetime objects
    dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])
    # Generate Sundays for the duration of the dataset
    sundays = pd.date_range(start=dataframe['timestamp'].min(), end=dataframe['timestamp'].max(), freq='W-SUN').strftime('%Y-%m-%d').tolist()
    # Add Sundays to the holiday list
    holidays_and_sundays = all_holidays + sundays
    # Add a new column to indicate holidays
    dataframe['is_holiday'] = dataframe['timestamp'].dt.strftime('%Y-%m-%d').isin(holidays_and_sundays).astype(int)
    # Return the processed dataframe
    return dataframe

# Process both train and test datasets
train_df_processed = process_dataset(train_df)
test_df_processed = process_dataset(test_df)

# Save the processed datasets
train_df_processed.to_csv('processed_train.csv', index=False)
test_df_processed.to_csv('processed_test.csv', index=False)