import subprocess
import datetime
import requests
import json
from dateutil import parser as dtparser
import math

PROM_URL = "http://localhost:9090"
QUERY = 'rate(openwhisk_action_activations_total[3m]) == 0'
NAMESPACE = "openwhisk"
IDLE_THRESHOLD_MINUTES = 3

# Optional: dry run (set to True to test without deleting)
DRY_RUN = True

SYSTEM_PREFIXES = (
    "openwhisk-",
    "kafka",
    "zookeeper",
    "couchdb",
    "redis",
    "nginx",
    "grafana",
    "prometheus",
    "controller",
    "invoker",
    "user-events",
    "alarmprovider",
    "apigateway",
    "wskadmin"
)

# Get all pods once
pods_output = subprocess.check_output([
    "kubectl", "get", "pods", "-n", NAMESPACE, "-o", "json"
]).decode()
pod_list = json.loads(pods_output)["items"]

# Filter pods once: exclude system pods
filtered_pod_list = [
    pod for pod in pod_list
    if not pod["metadata"]["name"].startswith(SYSTEM_PREFIXES)
]

# Get list of idle actions from Prometheus
res = requests.get(f"{PROM_URL}/api/v1/query", params={"query": QUERY})
results = res.json().get("data", {}).get("result", [])

for entry in results:
    action = entry["metric"].get("action")
    if not action:
        continue
    normalized_action = action.replace("-", "")
    ##this is setting for request rate and capacity per pod so that we estimate how many pod need
    request_rate = float(entry["value"][1]) #assume value[1] hold rate
    capacity_per_pod = 1.0 #adjust as needed
    require_replicas = max(1, math.ceil(request_rate/capacity_per_pod))

    # Matching all pods for this action

    matching_pods = [ pod for pod in filtered_pod_list
        if normalized_action in pod["metadata"]["name"] and pod["metadata"]["name"].startswith("wsk")
    ]

    #Sort by age (oldest first)
    sorted_pods = sorted(
        matching_pods,
        key=lambda p: dtparser.parse(p["metadata"]["creationTimestamp"])
    )

    #Decide how many extra pods to delete
    excess = len(sorted_pods) - require_replicas
    if excess > 0:
        print(f"Action '{action}' has {len(sorted_pods)} pods, only {require_replicas} needed. Deleting {excess}...")

    for pod in sorted_pods[:excess]:
        name = pod["metadata"]["name"]
        creation_time = dtparser.parse(pod["metadata"]["creationTimestamp"])
        age_minutes = (datetime.datetime.utcnow() - creation_time.replace(tzinfo=None)).total_seconds() / 60.0

        if age_minutes > IDLE_THRESHOLD_MINUTES:
            print(f"Deleting excess idle pod: {name} (action: {action}, age: {int(age_minutes)} min)")

            if not DRY_RUN:
                subprocess.run(["kubectl", "delete", "pod", "-n", NAMESPACE, name])
            else:
                print(f"[Dry run] Would delete pod: {name}")
