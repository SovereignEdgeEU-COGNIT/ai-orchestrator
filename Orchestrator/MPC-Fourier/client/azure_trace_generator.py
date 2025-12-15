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
from prometheus.monitoring_function import PrometheusClient
print('Running...')

# Image path
img_path = "../frame_sample/3.jpg"
image = cv2.imread(img_path)

# Read and encode image to Base64
with open(img_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

prom_client = PrometheusClient()
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

# ⏱️ Set runtime limit (in seconds)
k = 60  #k mintues
MAX_RUNTIME = 600*k
start_time = time.time()
print("Replaying single-function trace...")
for i, delay in enumerate(trace):
    now = time.time()
    if now - start_time > MAX_RUNTIME:
        print(f"🛑 Stopping after {MAX_RUNTIME} seconds.")
        break
    warm_container = prom_client.get_warm_container_count("wskopenwhisk-invoker-00-.*-guest-objectdetection")
    trace_warm_containers.append((now, warm_container))
    future = executor.submit(send_request, i)
    futures.append(future)
    print(f'delay ={delay}')
    time.sleep(delay)  # ⏱ this is the inter-arrival time


# Wait for all responses
for future in as_completed(futures):
    req_id, status_code, response_data, response_time, send_time, end_time = future.result()
    response_time_str = f"{response_time:.4f} seconds" if response_time is not None else "N/A"
    center_point = response_data.get("center_point") if response_data else None
    print(f"Request {req_id} completed in {response_time_str}, data = {center_point} | Status: {status_code}")
    result = future.result()
    if result:  # Only append if result is not None
        results.append((req_id, status_code, response_time, send_time, end_time))  # Drop response_data


# Summary
print("✅ All requests completed.")
if dropped_requests:
    print("\n❗ Dropped Requests Summary:")
    for log in dropped_requests:
        print(log)
else:
    print("\n✅ No requests were dropped.")

# Save results
df1 = pd.DataFrame(results, columns=[
    "RequestID", "StatusCode",  "ResponseTime", "SendTime", "EndTime"
])
df1.to_csv("../log/OpenWhisk_responsetime_tail_1_latest.csv", index=False)

df2 = pd.DataFrame(trace_warm_containers, columns=['Timestamp', 'total_container'])
df2.to_csv("../log/OpenWhisk__replicas_tail_1_latest.csv", index=False)

print("\n📁 Results saved to results_trace_Forecast_only.csv")
