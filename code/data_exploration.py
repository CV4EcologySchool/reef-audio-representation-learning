import json
from collections import Counter

# 1. Load the JSON data
json_path = r'/home/ben/reef-audio-representation-learning/data/dataset.json'
with open(json_path, 'r') as f: 
    data = json.load(f)

# 2. Extract the "data_type" values from the list of dictionaries
data_types = [entry["data_type"] for entry in data["audio"]]

# 3. Identify the unique entries for "data_type" and 
# 4. Count the number of occurrences for each unique "data_type" entry
data_type_counts = Counter(data_types)

# Extract unique data types, their counts, and lengths
unique_data_types = list(data_type_counts.keys())
count_occurrences = list(data_type_counts.values())

# Print the results
for dtype, count in zip(unique_data_types, count_occurrences):
    print(f"'{dtype}' has {count} occurrences.")

