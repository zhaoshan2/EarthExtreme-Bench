from collections import Counter

# Initialize a counter for the dates
date_counter = Counter()

# Read the file and parse the dates
with open("../filters/crops_daily_txt/imerg_rain_perc95_2023_3days.txt", "r") as file:
    for line in file:
        # Extract the list of dates from the line
        dates_str = line.split("Dates: ")[1].strip()
        dates = eval(
            dates_str
        )  # Convert the string representation of the list to an actual list
        # Update the counter with the dates
        date_counter.update(dates)

# Print the count of each date
for date, count in date_counter.items():
    print(f"Date: {date}, Count: {count}")

# Initialize a counter for the months
month_counter = Counter()

# Count the occurrences of each month
for date in date_counter:
    # Extract the month in 'YYYYMM' format
    month = date[:6]
    # Update the month counter
    month_counter.update([month])

# Print the count of each month
for month, count in month_counter.items():
    print(f"Month: {month}, Count: {count}")
