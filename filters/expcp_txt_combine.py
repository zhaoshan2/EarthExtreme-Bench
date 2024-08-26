from collections import defaultdict
from datetime import datetime, timedelta


# Function to read and parse the file contents
def parse_data(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            # Convert the string representation of tuples to actual tuples
            date_str, lat, lon, lat_num, lon_num = eval(line.strip())
            data.append((date_str, lat, lon, lat_num, lon_num))
    return data


# Helper function to check if a list of dates contains at least 3 consecutive days
def find_consecutive_days(dates):
    # Convert the string dates to datetime objects
    date_objs = sorted([datetime.strptime(date, "%Y%m%d") for date in dates])
    consecutive_dates = []

    # Find sequences of at least 3 consecutive dates
    for i in range(len(date_objs) - 2):
        if date_objs[i + 1] == date_objs[i] + timedelta(days=1) and date_objs[
            i + 2
        ] == date_objs[i] + timedelta(days=2):
            # Collect the consecutive dates
            j = i
            current_sequence = []
            while j < len(date_objs) and (
                date_objs[j] == date_objs[i] + timedelta(days=(j - i))
            ):
                current_sequence.append(date_objs[j].strftime("%Y%m%d"))
                j += 1
            consecutive_dates.append(current_sequence)

    # Flatten the list of lists and remove duplicates
    return sorted(set([date for seq in consecutive_dates for date in seq]))


# Function to filter lat-lon combinations lasting at least 3 days
def filter_lat_lon(data):
    lat_lon_dates = defaultdict(set)

    # Group dates by lat-lon combination
    for date_str, lat, lon, _, _ in data:
        lat_lon_dates[(lat, lon)].add(date_str)

    # Filter based on the number of unique dates
    filtered_lat_lon = {
        key: find_consecutive_days(dates)
        for key, dates in lat_lon_dates.items()
        if len(find_consecutive_days(dates)) >= 3
    }

    return filtered_lat_lon


# Main function to run the filtering process
def main():
    total_results = []
    year = 2023
    for month in range(1, 13):

        file_path = (
            f"../filters/crops_daily_txt/imerg_rain_perc95_{year}_month{month:02}.txt"
        )

        # Parse the data
        data = parse_data(file_path)

        # Filter lat-lon combinations lasting at least 3 days
        result = filter_lat_lon(data)
        total_results.append(result)

        # # Print the result
        # for (lat, lon), dates in result.items():
        #     print(f"Lat: {lat}, Lon: {lon}, Dates: {dates}")

    with open(
        f"../filters/crops_daily_txt/imerg_rain_perc95_{year}_3days.txt", "w"
    ) as f:
        for item in total_results:
            for (lat, lon), dates in item.items():
                f.write(f"Lat: {lat}, Lon: {lon}, Dates: {dates}\n")


if __name__ == "__main__":
    main()
