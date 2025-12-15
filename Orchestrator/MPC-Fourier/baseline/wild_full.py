import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import time
import requests
import base64
import pandas as pd
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from prometheus.monitoring_function import PrometheusClient
from Reclaim.final_clean_pod_reclaim import reclaim_pods
histograms = {}
invocation_history = {}
active_containers = {}
cutoff_percentile = 80
cold_starts = 0
futures = []

func = "objectdetection"  # assuming a single function for now
print('Running...')

# Image path
img_path = "../frame_sample/3.jpg"
image = cv2.imread(img_path)

# Read and encode image to Base64
with open(img_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

prom_client = PrometheusClient()
# Define the OpenWhisk Endpoint

whisk_url = "https://XXX.XXX.XX.XXX:XXXXX/api/v1/namespaces/guest/actions/object-detection?blocking=true&result=true"

#Authentication
auth_key = "FILL YOUR AUTH KEY"

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
num_requests = 50  # Change this value to control the load
# Store logs for dropped requests
dropped_requests = []
results = []
# Function to send a request and measure response time
executor = ThreadPoolExecutor(max_workers= 100)  # adjust parallelism
futures = []
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



med_trace_list = [249, 757, 1385, 1489, 1717, 1721]
tail_trace_list = [1,9,15,18,19,21,22]
item = tail_trace_list[0]
# Load one trace file
with open("../main_traces/tail_"+str(item)+".txt") as f:
    trace = [float(line.strip()) * 7 for line in f if line.strip()]

executor = ThreadPoolExecutor(max_workers=64)  # adjust if needed
results = []

trace_warm_containers = []

# Set runtime limit (in seconds)
k = 60  #k mintues
MAX_RUNTIME = 600*k
start_time = time.time()
print("Replaying single-function trace...")
for i, delay in enumerate(trace):
    now = time.time()
    if now - start_time > MAX_RUNTIME:
        print(f"Stopping after {MAX_RUNTIME} seconds.")
        break

    # Cold start detection
    if func not in invocation_history:
        invocation_history[func] = []
    if invocation_history[func]:
        idle_time = now - invocation_history[func][-1]
    else:
        idle_time = 0
    invocation_history[func].append(now)

    if func not in active_containers or active_containers[func] < now:
        cold_starts += 1
        print(f"[Wild] Cold start predicted for {func} at {now:.2f}")

    # Histogram update
    if func not in histograms:
        histograms[func] = []
    if idle_time > 0:
        histograms[func].append(idle_time)

    # Estimate keep-alive deadline
    if len(histograms[func]) >= 3:
        cutoff = np.percentile(histograms[func], cutoff_percentile)
        keep_alive_duration = cutoff
        source ="hist"
    else:
        try:
            history = invocation_history[func]
            model = ARIMA(history, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast()[0]
            keep_alive_duration = max(forecast - now, 60)
            source = "arima"
        except Exception:
            keep_alive_duration = 60
            source = "fallback"

    active_containers[func] = now + keep_alive_duration
    # Reclaim if any containers are past their keep_alive deadline
    reclaimable_funcs = []
    now = time.time()

    for f in list(active_containers):
        if now > active_containers[f]:
            reclaimable_funcs.append(f)

    if reclaimable_funcs:
        print(f"Found expired containers for: {reclaimable_funcs}")
        deleted = reclaim_pods(top_r=1, already_deleted_pods=[])
        print(f"Reclaimed pods: {deleted}")

    # Query warm container state from Prometheus
    warm_container = prom_client.get_warm_container_count("wskopenwhisk-invoker-00-.*-guest-objectdetection")
    trace_warm_containers.append((now, warm_container, keep_alive_duration, source))

    # Send the real request
    future = executor.submit(send_request, i)
    futures.append(future)

    print(f"delay = {delay:.2f}")
    time.sleep(delay)

# Wait for completion
for future in as_completed(futures):
    result = future.result()
    if result:
        req_id, status_code, response_data, response_time, send_time, end_time = result
        center_point = response_data.get("center_point") if response_data else None
        print(f"Request {req_id} completed in {response_time:.4f}s, data = {center_point} | Status: {status_code}")
        results.append((req_id, status_code, response_time, send_time, end_time))

# Save logs
df1 = pd.DataFrame(trace_warm_containers, columns=["Timestamp", "WarmContainers", "KeepAliveDuration", "PredictionSource"])

df1.to_csv("wild_predicted_containers.csv", index=False)
print(f"Saved warm container trace as 'wild_predicted_containers.csv'")

df2 = pd.DataFrame(results, columns=["RequestID", "StatusCode", "ResponseTime", "SendTime", "EndTime"])
df2.to_csv("wild_response_logs.csv", index=False)
print(f"Saved request logs as 'wild_response_logs.csv'")

print(f"Total predicted cold starts: {cold_starts}")
