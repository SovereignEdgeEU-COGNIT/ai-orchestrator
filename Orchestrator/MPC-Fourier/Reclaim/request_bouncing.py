import requests
import re
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd

# Loki setup
LOKI_URL = "http://localhost:3100"
QUERY = '{namespace="openwhisk", container="controller"} |= "posting activation to invoker"'


# Time window
# These must match the cluster time
START = int(datetime.strptime("2025-04-11T11:14:00", "%Y-%m-%dT%H:%M:%S").timestamp() * 1e9)
END = int(datetime.strptime("2025-04-11T13:57:00", "%Y-%m-%dT%H:%M:%S").timestamp() * 1e9)
# Query controller logs for "scheduled activation" and "posted to invoker"
query = '{namespace="openwhisk", container="controller"} |= "ShardingContainerPoolBalancer"'
params = {
    "query": query,
    "start": START,
    "end": END,
    "limit": 5000,
    "direction": "forward",
}

print("Fetching logs...")
resp = requests.get(f"{LOKI_URL}/loki/api/v1/query_range", params=params)
streams = resp.json()["data"]["result"]
if len(streams) == 0:
    print("No logs returned. You might need to expand the time range or increase the limit.")
# Store activationId -> threadId
activation_to_tid = {}
tid_post_counts = {}

print("Processing logs...")
for stream in streams:
    for ts_ns, log in stream["values"]:
        tid_match = re.search(r'\[#tid_([^\]]+)\]', log)
        tid = tid_match.group(1) if tid_match else None

        if "scheduled activation" in log:
            aid_match = re.search(r'scheduled activation ([a-f0-9]{32})', log)
            if aid_match and tid:
                activation_id = aid_match.group(1)
                activation_to_tid[activation_id] = tid
                tid_post_counts[tid] = 0  # init post count

        elif "posted to invoker" in log and tid:
            if tid in tid_post_counts:
                tid_post_counts[tid] += 1

# Build DataFrame
rows = []
for aid, tid in activation_to_tid.items():
    post_count = tid_post_counts.get(tid, 0)
    rows.append((aid, tid, post_count))

df = pd.DataFrame(rows, columns=["ActivationID", "ThreadID", "PostCount"])
df_sorted = df.sort_values(by="PostCount", ascending=False)

print("\nTop bouncing activations:")
print(df_sorted.tail(10))