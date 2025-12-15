import time
import random
import requests
import base64

import threading
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
stop_event = threading.Event()
# --- CONFIG ---
whisk_url = "https://130.239.48.225:31001/api/v1/namespaces/guest/actions/object-detection?blocking=true&result=true"
auth_key = "23bc46b1-71f6-4ed5-8c54-816aa4f8c502:123zO3xZCLrMN6v2BKK1dXYFpXlPkccOFqm12CdAsMgRU4VrNZ9lyGVCGuMDGIwP"

# --- Setup ---
header = {
    "Authorization": f"Basic {base64.b64encode(auth_key.encode()).decode()}",
    "Content-Type": "application/json"
}

# Preload and encode image
img_path = "../frame_sample/3.jpg"
with open(img_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

payload = {
    "object_of_interest": "red",
    "frame": encoded_image
}

# --- CSV Logging Setup ---
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = f"gradual_ramp_up_request_log_{timestamp_str}.csv"
with open(LOG_FILE, "w", newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["request_id", "timestamp", "scheduled_timestamp", "delay_from_schedule", "response_time", "status_code", "traffic_type"])
    writer.writeheader()

# Configuration
BURST_DURATION_RANGE = (1, 5)           # seconds
IDLE_DURATION_RANGE = (50, 800)         # seconds
MIN_RPS = 5
MAX_RPS = 300
NUM_BURSTS = 50                         # total number of bursts for the simulation

def send_requests(rps, duration):
    interval = 1.0 / rps
    end_time = time.time() + duration
    while time.time() < end_time:
        # Simulate sending a request
        print(f"Sent request at RPS={rps:.2f}")
        time.sleep(interval)


def simulate_normal_traffic(mean_interval=2.0, std_dev=0.5):
    global request_count
    while not stop_event.is_set():
        inter_arrival = max(0.1, random.gauss(mean_interval, std_dev))  # avoid negatives
        time.sleep(inter_arrival)
        request_count += 1
        scheduled_time = time.time()
        threading.Thread(target=send_request, args=(request_count,"normal", scheduled_time)).start()

def send_request(i, traffic_tpye, scheduled_time):
    start = time.time()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(start))
    timestamp_schedule = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(scheduled_time))
    delay_from_schedule = start - scheduled_time
    try:
        response = requests.post(whisk_url, json=payload, headers=header, verify=False)
        latency = time.time() - start
        status_code = response.status_code
        print(f"[{i}] {status_code} in {latency:.3f}s")
    except Exception as e:
        latency = time.time() - start
        status_code = "ERROR"
        print(f"[{i}] Request failed: {e}")

# Log to file
    with open(LOG_FILE, "a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["request_id", "timestamp", "scheduled_timestamp", "delay_from_schedule", "response_time", "status_code", "traffic_type"])
        writer.writerow({
            "request_id": i,
            "timestamp": timestamp,
            "scheduled_timestamp":timestamp_schedule,
            "delay_from_schedule": delay_from_schedule,
            "response_time": f"{latency:.3f}",
            "status_code": status_code,
            "traffic_type": traffic_tpye
        })


request_count = 0
# Start background normal traffic
normal_traffic_thread = threading.Thread(target=simulate_normal_traffic, daemon=True)
normal_traffic_thread.start()
time.sleep(20)
# --- Burst loop ---
request_count = 0
for i in range(NUM_BURSTS):
    # Gradually increase RPS over bursts
    rps = MIN_RPS + (i / (NUM_BURSTS - 1)) * (MAX_RPS - MIN_RPS)

    # Optional: increase burst duration slightly or keep random
    burst_duration = random.uniform(*BURST_DURATION_RANGE)

    # Optional: shorten idle duration over time (can remove for fixed idle)
    idle_duration = random.uniform(*IDLE_DURATION_RANGE)

    total_requests = int(rps * burst_duration)
    delay = 1.0 / rps
    burst_start = time.time()

    print(f"\n🔥 Burst: duration={burst_duration:.1f}s, RPS={rps}, total={total_requests}")

    with ThreadPoolExecutor(max_workers=int(rps * 1.5)) as executor:
        for i in range(total_requests):
            scheduled_time = burst_start + i * delay
            sleep_time = max(0, scheduled_time - time.time())
            time.sleep(sleep_time)

            request_count += 1
            traffic_type = f"attack_{burst_duration:.1f}s_{int(rps)}rps"
            executor.submit(send_request, request_count, traffic_type, scheduled_time)

    time.sleep(idle_duration)
print("✅ Simulation finished!")