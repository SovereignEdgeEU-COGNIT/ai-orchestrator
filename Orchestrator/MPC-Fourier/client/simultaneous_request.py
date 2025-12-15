import requests
import base64
import cv2
import time
import urllib3
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # or 'QtAgg' or 'Agg' depending on your setup
import matplotlib.pyplot as plt
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from concurrent.futures import ThreadPoolExecutor, as_completed

print('Running...')

# Image path
img_path = "../frame_sample/3.jpg"
image = cv2.imread(img_path)

# Read and encode image to Base64
with open(img_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')


# Define the OpenWhisk Endpoint
#whisk_url = "https://129.192.71.168:31001/api/v1/namespaces/guest/actions/object-detection?blocking=true&result=true"

whisk_url = "https://130.239.48.225:31001/api/v1/namespaces/guest/actions/object-detection?blocking=true&result=true"

#Authentication
auth_key = "23bc46b1-71f6-4ed5-8c54-816aa4f8c502:123zO3xZCLrMN6v2BKK1dXYFpXlPkccOFqm12CdAsMgRU4VrNZ9lyGVCGuMDGIwP"

header = {
    "Authorization": f"Basic {base64.b64encode(auth_key.encode()).decode()}",
    "Content-Type": "application/json"
}


# Define payload
payload = {
    "object_of_interest": "yellow",
    "frame": encoded_image
}

# Number of requests to send
num_requests = 1000  # Change this value to control the load
# Store logs for dropped requests
dropped_requests = []

# Function to send a request and measure response time

def send_request(req_id):
    send_time = time.time()
    try:
        response = requests.post(whisk_url, json=payload, headers=header, verify=False)
        end_time = time.time()
        response_time = end_time - send_time
        status_code = response.status_code

        if status_code == 503:
            dropped_requests.append(f"Request {req_id} - Dropped: 503 Service Unavailable")
        elif status_code != 200:
            dropped_requests.append(f"Request {req_id} - Failed: {status_code}, Response: {response.text}")

        return req_id, status_code, response.json(), response_time, send_time, end_time

    except requests.exceptions.Timeout:
        end_time = time.time()
        response_time = end_time - send_time
        dropped_requests.append(f"Request {req_id} - Timeout after {response_time:.2f}s")
        return req_id, "Timeout", None, response_time, send_time, end_time

    except requests.exceptions.RequestException as e:
        end_time = time.time()
        response_time = end_time - send_time
        dropped_requests.append(f"Request {req_id} - Error: {str(e)}")
        return req_id, "Error", None, response_time, send_time, end_time


t=0
last_execution_time = [None]  # Mutable variable to store execution timestamps
results = []  # ✅ Collect response data here
# Use ThreadPoolExecutor for parallel requests
with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers based on system resources

    futures = {executor.submit(send_request, i): i for i in range(num_requests)}

    for future in as_completed(futures):
        req_id, status_code, response_data, response_time, send_time, end_time = future.result()
        response_time_str = f"{response_time:.4f} seconds" if response_time is not None else "N/A"
        center_point = response_data.get("center_point") if response_data else None
        print(f"Request {req_id} completed in {response_time_str}, data = {center_point}  | Status: {status_code}")

        if status_code == 200:
            t += 1
        if response_time is not None:
            results.append((req_id, status_code, response_time, send_time, end_time))


print("All requests completed.")
# Print summary of dropped requests
if dropped_requests:
    print("\nDropped Requests Summary:")
    for log in dropped_requests:
        print(log)
else:
    print("\nNo requests were dropped!")

print(t)

##

# Convert to DataFrame
df = pd.DataFrame(results, columns=["RequestID", "StatusCode", "ResponseTime", "SendTime", "EndTime"])
df.sort_values(by="SendTime", inplace=True)
df["TimeBetweenRequests"] = df["SendTime"].diff().fillna(0)

print("\n📊 Response Time Percentiles:")
for p in [0, 25, 50, 75, 90, 95, 99, 100]:
    value = df["ResponseTime"].quantile(p / 100)

    if p == 0:
        label = "Min"
        icon = "🟢"
    elif p == 50:
        label = "Median"
        icon = "⚪"
    elif p == 100:
        label = "Max"
        icon = "🔴"
    elif p >= 90:
        label = f"{p}th (slow)"
        icon = "🐢"
    elif p <= 25:
        label = f"{p}th (fast)"
        icon = "⚡"
    else:
        label = f"{p}th"
        icon = "➖"

    print(f"{icon} {label:12}: {value:.4f} seconds")

# Plot response time of successful requests
plt.figure(figsize=(10, 6))
plt.boxplot(df["ResponseTime"], vert=True, patch_artist=True, tick_labels=["200 OK"])

plt.title("Response Time Distribution for Successful Requests")
plt.ylabel("Response Time (seconds)")
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df["ResponseTime"], bins=20, edgecolor='black', color='skyblue')

plt.title("Histogram of Response Times (200 OK)")
plt.xlabel("Response Time (seconds)")
plt.ylabel("Number of Requests")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 5))
df = pd.DataFrame(results, columns=["RequestID", "StatusCode", "ResponseTime", "SendTime", "EndTime"])
df.sort_values("RequestID", inplace=True)
plt.scatter(df["RequestID"], df["ResponseTime"], label="Response Time")
plt.axhline(y=0.28, color='green', linestyle='--', label='Average Execution Time')
#plt.axhline(y=1.2, color='red', linestyle='--', label='Queuing Threshold?')
plt.xlabel("Request ID")
plt.ylabel("Response Time (s)")
plt.title("OpenWhisk Response Times")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Plot
# Define your average execution time threshold
average_execution_time = 0.3
#df['WarmContainers'] = [1,1,2,2,3]
df.to_csv("../log/results_synthetic_rate_10_1000.csv", index=False)
# Create masks
import matplotlib.dates as mdates
from datetime import datetime
# 🔁 Convert UNIX timestamps to datetime.datetime objects
df["SendTime"] = df["SendTime"].apply(lambda ts: datetime.fromtimestamp(ts))
df.sort_values("SendTime", inplace=True)

# Define threshold
average_execution_time = 0.29
below_avg = df["ResponseTime"] <= average_execution_time
above_avg = df["ResponseTime"] > average_execution_time

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(df[below_avg]["SendTime"], df[below_avg]["ResponseTime"], label="≤ Avg Time", color="blue")
plt.scatter(df[above_avg]["SendTime"], df[above_avg]["ResponseTime"], label="> Avg Time", color="red")
plt.axhline(y=average_execution_time, color='green', linestyle='--', label='Avg Exec Time (0.4s)')

# ✅ Format x-axis labels as HH:MM:SS safely
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gcf().autofmt_xdate()

plt.xlabel("Send Time (HH:MM:SS)")
plt.ylabel("Response Time (s)")
plt.title("OpenWhisk Response Times Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


##
def apply_latex_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

apply_latex_style()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot 1: Response Time
ax1.scatter(df[below_avg]["SendTime"], df[below_avg]["ResponseTime"], label="≤ Avg Time", color="blue")
ax1.scatter(df[above_avg]["SendTime"], df[above_avg]["ResponseTime"], label="> Avg Time", color="red")
ax1.axhline(y=average_execution_time, color='green', linestyle='--', label='Avg Exec Time')
ax1.set_ylabel("Response Time (s)")
ax1.set_title("Response Time per Request")
ax1.legend()
ax1.grid(True)
import matplotlib.ticker as ticker
# Plot 2: Warm Containers
ax2.plot(df["SendTime"], df["WarmContainers"], color="orange", marker='o')
ax2.set_ylabel("Warm Containers")
ax2.set_xlabel("Send Time")
ax2.set_title("Warm Container Count Over Time")
ax2.grid(True)

# Ensure only integer ticks on y-axis
ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Format x-axis as hh:mm:ss
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
fig.autofmt_xdate()

plt.tight_layout()
plt.show()