import requests
import re
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have it installed

import matplotlib.pyplot as plt
# your plotting code here


LOKI_URL = "http://localhost:3100"

NAMESPACE = "openwhisk"
CONTAINER = "invoker"

# Time window
# These must match the cluster time
START = int(datetime.strptime("2025-04-11T19:46:00", "%Y-%m-%dT%H:%M:%S").timestamp() * 1e9)
END = int(datetime.strptime("2025-04-11T20:50:00", "%Y-%m-%dT%H:%M:%S").timestamp() * 1e9)
print("Fetching logs from Loki...")


def fetch_logs(filter_string):
    print(f"Querying logs for: {filter_string}")
    response = requests.get(
        f"{LOKI_URL}/loki/api/v1/query_range",
        params={
            "query": f'{{namespace="{NAMESPACE}", container="{CONTAINER}"}} |= "{filter_string}"',
            "start": int(START),
            "end": int(END),
            "limit": 5000,
            "direction": "forward"
        }
    )
    return response.json()["data"]["result"]


# Fetch both logs
init_streams = fetch_logs("containerStart")
ack_streams = fetch_logs("posted completion of activation")



init_times = {}
ack_times = {}

for stream in init_streams:
    for ts_ns, log in stream["values"]:
        ts = datetime.utcfromtimestamp(int(ts_ns) / 1e9)

        # Match containerStart lines with activationId
        match_init = re.search(r"containerStart.*activationId: ([a-f0-9]{32})", log)
        if match_init:
            activation_id = match_init.group(1)
            init_times[activation_id] = ts


    # Parse ack logs
for stream in ack_streams:
    for ts_ns, log in stream["values"]:
        ts = datetime.utcfromtimestamp(int(ts_ns) / 1e9)
        match = re.search(r"activation ([a-f0-9]{32})", log)
        if match:
            ack_times[match.group(1)] = ts

rows = []
for aid in init_times:
    if aid in ack_times:
        delay = (ack_times[aid] - init_times[aid]).total_seconds()
        rows.append((aid, init_times[aid], ack_times[aid], delay))

df = pd.DataFrame(rows, columns=["ActivationID", "InitTime", "AckTime", "Duration"])
print(min(df["Duration"]))
if df.empty:
    print("⚠️ No matching activation logs found in time window.")
else:
    print(df.sort_values(by="Duration", ascending=False).tail(1000))
    df["Duration"].hist(bins=20, grid=True, color="skyblue", edgecolor="black")
    plt.xlabel("Wait Duration (s)")
    plt.ylabel("Count")
    plt.title("Activation Wait Time Distribution")
    plt.show()
