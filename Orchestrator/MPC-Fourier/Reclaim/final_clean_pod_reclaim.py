import requests
import time
from datetime import datetime
from kubernetes import client, config
from prometheus.monitoring_function import  KubectlPodRanker
from .reclaim_cache import is_recently_deleted, update_recent_deletions, cleanup_deletion_history

# --- Configuration ---

LOKI_URL = "http://localhost:3100/loki/api/v1/query"
NAMESPACE = "openwhisk"
TIMEOUT_SECONDS = 300  # max waiting time
POLL_INTERVAL = 2  # how often to check Loki
DRAIN_LABEL = {"metadata": {"labels": {"drain": "true"}}}


# --- Kubernetes Setup ---

def init_kubernetes():
    config.load_kube_config()
    return client.CoreV1Api()


# --- Step 1: Label Pod as drain=true ---

def label_pod_drain(v1, pod_name):
    v1.patch_namespaced_pod(name=pod_name, namespace=NAMESPACE, body=DRAIN_LABEL)
    print(f"Labeled pod {pod_name} with drain=true.")


# --- Step 2: Query Loki for activations running on this pod ---

def query_loki(query, window_seconds = 700): #window_seconds is the time in past to query it is in second
    now = int(time.time() *1e9) #nanosecond
    start = now - int(window_seconds *1e9) # nanosecond ago
    params = {
        "query": query,
        "start": str(start),
        "end": str(now),
        "limit": 1000
    }
    response = requests.get(LOKI_URL, params=params)
    response.raise_for_status()
    return response.json()


def get_tids_for_pod(pod_name, function_path="/guest/object-detection"):
    #print(f"Getting TIDs for pod {pod_name} and function {function_path}...")
    query = f'{{app="openwhisk-openwhisk"}} |= "sending arguments to {function_path} at ContainerId({pod_name})"'
    result = query_loki(query)
    tids = set()
    for stream in result['data']['result']:
        for value in stream['values']:
            logline = value[1]
            start = logline.find("[#tid_")
            if start != -1:
                end = logline.find("]", start)
                tid = logline[start + 6:end]
                tids.add(tid)
    #print(f"Found {len(tids)} TIDs.")
    return list(tids)



def get_activation_ids_from_tids(tids):
    #print("Getting Activation IDs from TIDs...")
    activation_ids = set()
    for tid in tids:
        query = f'{{app="openwhisk-openwhisk"}} |= "{tid}" |= "invoker_activation_start"'
        result = query_loki(query)
        for stream in result['data']['result']:
            for value in stream['values']:
                logline = value[1]
                # Parse Activation ID from logline (activationId: XXXXX)
                marker = "activationId:"
                idx = logline.find(marker)
                if idx != -1:
                    activation_id = logline[idx + len(marker):].strip().split()[0]
                    activation_ids.add(activation_id)
    #print(f"Found {len(activation_ids)} activation IDs.")
    return list(activation_ids)


# --- Step 3: Wait for Activations to Complete ---

def wait_for_activations_to_complete(activation_ids):
    #print(f"Waiting for {len(activation_ids)} activations to complete...")
    unfinished = set(activation_ids)
    start_time = time.time()

    while unfinished and time.time() - start_time < TIMEOUT_SECONDS:
        for activation_id in list(unfinished):
            query = f'{{app="openwhisk"}} |= "{activation_id}" |= "MessagingActiveAck"'
            result = query_loki(query)
            if result['data']['result']:
                #print(f"Activation {activation_id} finished.")
                unfinished.remove(activation_id)
        if unfinished:
            time.sleep(POLL_INTERVAL)

    if unfinished:
        print(f"Timeout: {len(unfinished)} activations did not finish.")
    else:
        print("All activations completed.")


# --- Step 4: Delete the Pod ---

def delete_pod(v1, pod_name):
    v1.delete_namespaced_pod(name=pod_name, namespace=NAMESPACE)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Pod {pod_name} deleted.")


# --- Main Logic ---

def drain_and_reclaim_pod(v1, pod_name):

    label_pod_drain(v1, pod_name)

    tids = get_tids_for_pod(pod_name)
    activation_ids = get_activation_ids_from_tids(tids)

    if activation_ids:
        wait_for_activations_to_complete(activation_ids)

    delete_pod(v1, pod_name)

def drain_and_reclaim_pod_v2(v1, pod_name):
    # 1. Reclaim the container inside Invoker
    reclaim_url = "http://localhost:18081/reclaim"
    headers = {"Content-Type": "application/json"}
    reclaim_data = {"containerId": pod_name}  # NOTE: The containerId must match exactly what Invoker expects

    try:
        resp = requests.post(reclaim_url, json=reclaim_data, headers=headers)
        resp.raise_for_status()
        print(f"[OK] Sent reclaim request to Invoker for pod {pod_name}: {resp.text}")
    except Exception as e:
        print(f"[ERROR] Failed to send reclaim request to Invoker: {e}")
        return

    # 2. Wait briefly to allow Invoker to update its internal state
    time.sleep(2)
    # 3. Now delete the Kubernetes pod
    try:
        delete_options = client.V1DeleteOptions(grace_period_seconds=0)
        v1.delete_namespaced_pod(name=pod_name, namespace=namespace, body=delete_options)
        print(f"[OK] Deleted pod {pod_name} from Kubernetes.")
    except Exception as e:
        print(f"[ERROR] Failed to delete pod {pod_name}: {e}")

def list_function_pods(v1, label_selector="app=openwhisk"):
    pods = v1.list_namespaced_pod(namespace=NAMESPACE, label_selector=label_selector).items
    pod_names = [pod.metadata.name for pod in pods]
    return pod_names


# --- Run ---
def main(top_r=0):
    delete_pods=[]
    ranker = KubectlPodRanker(
        namespace="openwhisk",
        loki_url=LOKI_URL,
        function_keywords=["guest", "objectdetection"]
    )


    top_pods = ranker.rank_pods(top_r=top_r)

    if len(top_pods) == 0:
        print('There is no pod available')
    else:

        #print(f"\nTop {top_r} pods to drain:")
        #for pod in top_pods:
        #    print(f"  {pod['pod_name']} (score={pod['score']:.3f})")

        top_pods_name_list = [pod['pod_name'] for pod in top_pods]
        # initialize Kubernetes API
        v1 = init_kubernetes()

        # get list of running pods
        running_pods = ranker.list_running_function_pods(v1)

        # filter
        safe_top_pods = [pod for pod in top_pods_name_list if pod in running_pods]

        print("\nReady to drain these pods:")
        for pod in safe_top_pods:
            print(f"  {pod}")
            drain_and_reclaim_pod(v1, pod)
def reclaim_pods(top_r, already_deleted_pods):
    deleted_pods = []
    ranker = KubectlPodRanker(
        namespace="openwhisk",
        loki_url=LOKI_URL,
        function_keywords=["guest", "objectdetection"]
    )


    top_pods = ranker.rank_pods(top_r=top_r, already_deleted_pods= already_deleted_pods)

    if len(top_pods)==0:
        print('There is no pod available')
        return []
    else:

        print(f"\nTop {top_r} pods to drain:")
        for pod in top_pods:
            print(f"  {pod['pod_name']} (score={pod['score']:.3f})")

        top_pods_name_list = [pod['pod_name'] for pod in top_pods]
        # initialize Kubernetes API
        v1 = init_kubernetes()

        # get list of running pods
        running_pods = ranker.list_running_function_pods(v1)

        # filter
        safe_top_pods = [pod for pod in top_pods_name_list if pod in running_pods]

        print("\nReady to drain these pods:")
        for pod in safe_top_pods:
            if is_recently_deleted(pod):
                print(f"Skip reclaiming {pod} — already marked for deletion recently.")
                continue
            #print(f"  {pod}")
            drain_and_reclaim_pod(v1, pod)
            deleted_pods.append(pod)

        return deleted_pods




