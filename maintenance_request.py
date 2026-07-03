import csv
import os
from datetime import datetime

# Ask the user for maintenance times
print("Maintenance allocation:")
today = datetime.now()
start_hour = int(input("Enter start hour: "))
end_hour   = int(input("Enter end hour: "))
start_datetime = today.replace(hour=start_hour, minute=0, second=0, microsecond=0)
end_datetime   = today.replace(hour=end_hour,   minute=0, second=0, microsecond=0)

# Prepare the row
row = {'start_datetime': start_datetime, 'end_datetime': end_datetime}

# Path to your CSV
path = 'maintenance.csv'

# Check if file exists and is non‑empty
write_header = not os.path.exists(path) or os.path.getsize(path) == 0

# Open in append mode
with open(path, 'a', newline='') as csvfile:
    fieldnames = ['start_datetime', 'end_datetime']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write header only if needed
    if write_header:
        writer.writeheader()

    writer.writerow(row)

print(f"Appended maintenance window to {path}.")


    