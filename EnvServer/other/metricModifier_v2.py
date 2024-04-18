import pandas as pd
import json

# Assuming csv_file_path and txt_file_path are the paths to the CSV and TXT files respectively
csv_file_path = "./exportedMetrics.csv"
txt_file_path = "./load_gen_data.txt"

# Load the CSV file
csv_df = pd.read_csv(csv_file_path)

# Filter out rows where 'id' ends with '_'
csv_df = csv_df[~csv_df['id'].str.endswith('_')]

#print(csv_df.head())

# Convert memory to 'Memory usage [KB]'
csv_df['Memory usage [KB]'] = csv_df['memory'] // 1024

# Change timestamp format to long
#csv_df['ts'] = pd.to_datetime(csv_df['ts']).astype(int) #/ 10**6

# Rename columns as per requirement
csv_df.rename(columns={'ts': 'Timestamp [ms]',
                       'cpu': 'CPU usage [%]',
                       'disk_read': 'Disk read throughput [KB/s]',
                       'disk_write': 'Disk write throughput [KB/s]',
                       'net_rx': 'Network received throughput [KB/s]',
                       'net_tx': 'Network transmitted throughput [KB/s]'},
              inplace=True)

# Load and parse the TXT file
host_info_mapping = {}
cognit1 = 1024
cognit2 = 1024
with open(txt_file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        prefix = "Cognit-test" if data['host_info']['name'] == "Cognit-test" else "Cognit-test-2"
        name = prefix + "_emu_sr_" + data['client_info']['flavor'] + "_"
        if data['host_info']['name'] == "Cognit-test":
            suffix = str(cognit1)
            cognit1 += 1
        else:
            suffix = str(cognit2)
            cognit2 += 1
        name += suffix
        host_info_mapping[name] = {'CPU cores': data['sr_env']['cpu'], 'Memory capacity provisioned [KB]': data['sr_env']['mem'] * 1024, 'flavor': data['client_info']['flavor']}

# Convert the host_info_mapping dictionary to a DataFrame
host_info_df = pd.DataFrame.from_dict(host_info_mapping, orient='index').reset_index().rename(columns={'index': 'id'})
#print(host_info_df)

# Assuming there's a way to merge csv_df with host_info_df (e.g., using 'id' or another suitable column after adjustment)
# For demonstration, it's assumed that the 'id' in csv_df can be matched with 'id' in host_info_df
# This step might need adjustment based on how 'id' values correlate between CSV data and TXT data

# Merge CSV DataFrame with host info DataFrame
merged_df = pd.merge(csv_df, host_info_df, on='id', how='left')

# Select and rearrange columns
merged_df = merged_df[['id', 'flavor', 'Timestamp [ms]', 'CPU cores',  'CPU usage [%]', 'Memory capacity provisioned [KB]', 'Memory usage [KB]', 'Disk read throughput [KB/s]',
                 'Disk write throughput [KB/s]', 'Network received throughput [KB/s]',
                 'Network transmitted throughput [KB/s]']]

# Save the merged DataFrame to a new CSV file
#merged_df.to_csv('merged_metrics.csv', index=False)

# Save to different files based on id
for id in merged_df['id'].unique():
    merged_df[merged_df['id'] == id].to_csv(id + '.csv', index=False)

