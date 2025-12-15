
import os
import redis
import time
import numpy as np
import json
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

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def posion():
    NUM_REQUESTS = 1000
    distribution = "poisson"  # or "exponential", "normal"
    rate = 10  # requests per second (lambda), used in Poisson/Exponential

    for i in range(NUM_REQUESTS):
        now = time.time()
        payload = {
            "req_id": i,
            "enqueue_time": now  # seconds since epoch
        }
        r.rpush("request_queue", json.dumps(payload))
        # Log timestamp separately
        r.rpush("request_timestamps", str(now))
        print(f"[Producer] Queued request {i}")

        if distribution == "poisson":
            interarrival = np.random.exponential(1 / rate)
        elif distribution == "exponential":
            interarrival = np.random.exponential(1 / rate)
        elif distribution == "normal":
            interarrival = max(0, np.random.normal(loc=1 / rate, scale=0.2))
        else:
            interarrival = 1  # fallback

        time.sleep(interarrival)


def azure():
    import redis
    import json
    import time
    import numpy as np
    from prometheus.monitoring_function import PrometheusClient
    import pandas as pd
    prom_client = PrometheusClient()

    r = redis.Redis(host='localhost', port=6379, decode_responses=True)

    # Load a trace file from Azure dataset
    med_trace_list = [249, 757, 1385, 1489, 1717, 1721]
    tail_trace_list = [1,9,15,18,19,21,22]
    item = tail_trace_list[0]

    #item = med_trace_list[0]  # Choose a trace
    with open("main_traces/tail_" + str(item) + ".txt") as f:
        trace = [float(line.strip()) * 7 for line in f if line.strip()]

    # Limit runtime
    k = 60  # k mintues
    MAX_RUNTIME = 600 * k

    start_time = time.time()
    print("Replaying trace as Redis requests...")

    results = []
    trace_warm_containers = []

    for i, delay in enumerate(trace):
        now = time.time()
        if now - start_time > MAX_RUNTIME:
            print(f"Stopping after {MAX_RUNTIME} seconds.")
            break
        warm_container = prom_client.get_warm_container_count("wskopenwhisk-invoker-00-.*-guest-objectdetection")
        trace_warm_containers.append((now, warm_container))

        enqueue_time = time.time()
        payload = {
            "req_id": i,
            "enqueue_time": enqueue_time
        }

        r.rpush("request_queue", json.dumps(payload))
        r.rpush("request_timestamps", str(enqueue_time))
        print(f"[Producer] Queued request {i} at {enqueue_time}")

        results.append((i, enqueue_time))
        time.sleep(delay)

    LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "log")
    os.makedirs(LOG_DIR, exist_ok=True)
    # Save timestamps to CSV
    df = pd.DataFrame(results, columns=["RequestID", "EnqueueTime"])
    df.to_csv(os.path.join(LOG_DIR, "MPC_results_trace_tail_1.csv"), index=False)
    print("\nResults saved to MPC_results_trace_med_249.csv")
    df2 = pd.DataFrame(trace_warm_containers, columns=['Timestamp', 'total_container'])
    df2.to_csv(os.path.join(LOG_DIR, "MPC_replicas_tail_1.csv"), index=False)



azure()



