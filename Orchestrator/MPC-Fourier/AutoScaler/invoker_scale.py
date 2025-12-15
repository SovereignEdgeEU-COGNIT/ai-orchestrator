import requests
import subprocess
import json
import time

# === Configs ===
PROM_URL = "http://localhost:9090"
DEPLOYMENT = "openwhisk-invoker"
NAMESPACE = "openwhisk"
MAX_REPLICAS = 10
THRESHOLD_WAIT = 0.3  # seconds
CHECK_INTERVAL = 5   # seconds between checks
SCALE_UP_STEP = 1     # how many replicas to add when scaling up

# === PromQL query ===
WAIT_QUERY = (
    'rate(openwhisk_action_waitTime_seconds_sum[1m]) / '
    'rate(openwhisk_action_waitTime_seconds_count[1m])'
)

def get_avg_wait_time():
    try:
        r = requests.get(f"{PROM_URL}/api/v1/query", params={"query": WAIT_QUERY})
        r.raise_for_status()
        results = r.json()["data"]["result"]
        values = [float(item["value"][1]) for item in results
                  if item["value"][1] not in ("NaN", "nan", None)
                  ]
        return max(values) if values else 0.0
    except Exception as e:
        print("Failed to fetch wait time:", e)
        return 0.0

def get_current_replicas():
    try:
        output = subprocess.check_output([
            "kubectl", "get", "statefulset", DEPLOYMENT,
            "-n", NAMESPACE, "-o", "json"
        ])
        deploy = json.loads(output)
        return deploy["spec"]["replicas"]
    except Exception as e:
        print("Error getting current replicas:", e)
        return None

def scale_invokers(new_count):
    try:
        subprocess.run([
            "kubectl", "scale", "statefulset", DEPLOYMENT,
            f"--replicas={new_count}", "-n", NAMESPACE
        ], check=True)
        print(f"Scaled to {new_count} replicas.")
    except subprocess.CalledProcessError as e:
        print("Failed to scale:", e)

def monitor_loop():
    print("Starting real-time autoscaler loop...")
    while True:
        wait_time = get_avg_wait_time()
        print(f"Average wait time: {wait_time:.2f}s")

        if wait_time > THRESHOLD_WAIT:
            current = get_current_replicas()
            if current is None:
                time.sleep(CHECK_INTERVAL)
                continue

            if current < MAX_REPLICAS:
                new_count = min(current + SCALE_UP_STEP, MAX_REPLICAS)
                print(f" High wait time! Scaling up from {current} to {new_count}")
                #scale_invokers(new_count)
            else:
                print(f"At max replicas ({MAX_REPLICAS}).")
        else:
            print(" Wait time within threshold. No action.")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    monitor_loop()
