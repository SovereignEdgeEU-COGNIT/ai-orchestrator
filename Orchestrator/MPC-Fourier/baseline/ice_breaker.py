# icebreaker_single_function.py
# Faithful adaptation of IceBreaker for homogeneous serverless + single function

import redis
import time
import json
import base64
import requests
import numpy as np
from threading import Thread
from numpy.fft import fft, fftfreq
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
from prometheus.monitoring_function import PrometheusClient
from Reclaim import final_clean_pod_reclaim
from utility import forecast_log
from AutoScaler import prewarm
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from Reclaim import final_clean_pod_reclaim

# Setup
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
session = requests.Session()
auth_key = "FILL YOUR AUTH"
header = {
    "Authorization": f"Basic {base64.b64encode(auth_key.encode()).decode()}",
    "Content-Type": "application/json"
}
whisk_url = "https://XXX.XXX:XXX/api/v1/namespaces/guest/actions/object-detection?blocking=true&result=true"
action_name = "object-detection"

img_path = "frame_sample/3.jpg"
with open(img_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# Parameters
H = 5
local_window = 10
L_cold = 10.0
L_warm = 0.28
mu = 1 / L_warm
warm_containers = 1
max_containers = 40
executor = ThreadPoolExecutor(max_workers=warm_containers)
sema = BoundedSemaphore(warm_containers)

# Request rate history
request_history = []
mem_w, cs_w, tn_w, fp_w = 0.25, 0.25, 0.25, 0.25

current_warm = 1  # or 1 to start
sema = BoundedSemaphore(current_warm)

def update_executor_if_needed(new_worker_count):
    global executor, warm_containers
    if new_worker_count != warm_containers:
        print(f"[INFO] Updating ThreadPoolExecutor to {new_worker_count} workers")
        executor.shutdown(wait=False)  # do not wait for existing tasks
        warm_containers = new_worker_count
        executor = ThreadPoolExecutor(max_workers=warm_containers)
def update_semaphore_if_needed(new_warm_count):
    global current_warm, sema
    if new_warm_count != current_warm:
        current_warm = new_warm_count
        sema = BoundedSemaphore(current_warm)
    print(f"[INFO] Updated semaphore to {current_warm} warm containers")

def fourier_forecast(x, n_predict=5, harmonics=5):
    n = len(x)
    t = np.arange(n)
    coeffs = np.polyfit(t, x, 2)
    trend = np.polyval(coeffs, t)
    detrended = x - trend
    fft_vals = fft(detrended)
    freqs = fftfreq(n)
    idxs = np.argsort(np.abs(freqs))
    t_future = np.arange(n, n + n_predict)
    restored = np.zeros(n_predict)
    for i in idxs[:1 + harmonics * 2]:
        amp = np.abs(fft_vals[i]) / n
        phase = np.angle(fft_vals[i])
        restored += amp * np.cos(2 * np.pi * freqs[i] * t_future + phase)
    forecast = restored + np.polyval(coeffs, t_future)
    return np.clip(forecast, 0, None)


def hybrid_controller(queue_len, warm_containers, forecast):
    pred = forecast[0]
    memory_cost = warm_containers
    cs_cost = max(0, pred - warm_containers) * L_cold
    tn = max(0, queue_len - pred)
    fp = max(0, pred - queue_len)
    score = mem_w * memory_cost + cs_w * cs_cost + tn_w * tn + fp_w * fp

    if score < 0.33:
        return 0
    elif score < 0.66:
        return int(round(pred))
    else:
        return min(int(round(pred + 1)), max_containers)


def maybe_cold_start(n):
    if n > 0:
        print(f"[ICEBREAKER] Prewarming {n} containers")
        prewarm.prewarm_with_shell(n, action_name)


def maybe_reclaim(n):
    if n > 0:
        print(f"[ICEBREAKER] Reclaiming {n} containers")
        final_clean_pod_reclaim.reclaim_pods(n)


def send_request(req, afterMPC=True):
    with sema:
        try:
            req_id = req["req_id"]
            enqueue_time = req["enqueue_time"]
            dequeue_time = time.time()
            queue_delay = dequeue_time - enqueue_time

            payload = {
                "object_of_interest": "red",
                "frame": encoded_image
            }
            if afterMPC:
                print(f"Sending request {req_id} | Queue delay: {queue_delay:.2f}s send after MPC")
            else:
                print(f"Sending request {req_id} | Queue delay: {queue_delay:.2f}s-- send before MPC")

            start_time = time.time()
            response = session.post(whisk_url, json=payload, headers=header, verify=False)
            end_time = time.time()

            http_latency = end_time - start_time
            total_delay = end_time - enqueue_time

            status = response.status_code
            try:
                resp_json = response.json()
                center = resp_json.get("center_point")
            except Exception:
                center = None

            print(f"Request {req_id} | Status: {status} | HTTP: {http_latency:.2f}s | Total: {total_delay:.2f}s | Center: {center}")
            with open("log_request_latency.csv", "a") as f:
                f.write(f"{req_id},{enqueue_time},{dequeue_time},{http_latency:.3f},{total_delay:.3f},{status}\n")

        except Exception as e:
            print(f"Request failed: {e}")

pod_name = "wskopenwhisk-invoker-00-.*-guest-objectdetection"
prom = PrometheusClient(ip="localhost", port=9090) # work with p06
# Main loop
request_history = []
trace_window = 10  # how many recent sample to keep for forecast
step_index = 0
while True:
    queue_len = r.llen("request_queue")
    print(f"[INFO] Queue length: {queue_len}")
    start = time.time()
    rate_window = r.lrange("request_timestamps", -30, -1)
    now = time.time()
    recent_times = [float(t) for t in rate_window if now - float(t) <= 30]
    arrival_rate = len(recent_times) / 30.0
    w = int(prom.get_warm_container_count(pod_name))
    request_history.append(arrival_rate)
    if len(request_history) > local_window:
        request_history.pop(0)

    if len(request_history) < local_window:
        print(
            f"[WARN] [{time.strftime('%H:%M:%S')}] Insufficient history ({len(request_history)} points), skipping MPC decision.")
        # Fallback: send all available requests without MPC guidance
        print(f"[Fallback] Sending all queued requests without MPC...")
        print(f'warm now ={w}')
        if w > 1:
            update_semaphore_if_needed(w)
            update_executor_if_needed(w)

        else:
            update_semaphore_if_needed(queue_len)
            update_executor_if_needed(max(w, queue_len))

        dispatched = 0
        while True:
            req_raw = r.lpop("request_queue")
            if not req_raw:
                break
            try:
                req = json.loads(req_raw)
                executor.submit(send_request, req)
                dispatched += 1
            except Exception as e:
                print(f"Dispatch error: {e}")
        print(f"[Fallback] Dispatched {dispatched} requests.")
        step_index += 5
        next_step_time = start + 5
        time_to_wait = next_step_time - time.time()
        if time_to_wait > 0:
            time.sleep(time_to_wait)
        continue  # skip this loop iteration

    forecast = fourier_forecast(np.array(request_history), n_predict=H)
    print(f"[INFO] Forecast: {[round(f, 2) for f in forecast]}")

    desired_warm = hybrid_controller(queue_len, warm_containers, forecast)
    print(f"[ICEBREAKER] Desired warm containers: {desired_warm}")

    if desired_warm > warm_containers:
        maybe_cold_start(desired_warm - warm_containers)
    elif desired_warm < warm_containers:
        maybe_reclaim(warm_containers - desired_warm)

    if desired_warm != warm_containers:
        executor.shutdown(wait=False)
        warm_containers = desired_warm
        executor = ThreadPoolExecutor(max_workers=warm_containers)
        sema = BoundedSemaphore(warm_containers)

    dispatched = 0
    for _ in range(warm_containers):
        req_raw = r.lpop("request_queue")
        if not req_raw:
            break
        req = json.loads(req_raw)
        executor.submit(send_request, req, True)
        dispatched += 1

    print(f"[DISPATCH] Sent {dispatched} requests.")
    time.sleep(5)
