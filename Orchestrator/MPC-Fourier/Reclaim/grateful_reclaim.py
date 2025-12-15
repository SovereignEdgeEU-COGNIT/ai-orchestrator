from kubernetes import client, config
import time
import requests
from prometheus.monitoring_function import  KubectlPodRanker
def label_pod_drain(namespace, pod_name):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    body = {
        "metadata": {
            "labels": {
                "drain": "true"
            }
        }
    }
    v1.patch_namespaced_pod(name=pod_name, namespace=namespace, body=body)
    print(f"Pod {pod_name} labeled as drain=true.")


def wait_for_activation_completion(loki_query, activation_id, timeout=60):
    # Your code to query Loki (HTTP query API) and check if
    # MessagingActiveAck for this activationId appears.
    # This part depends a bit on your Loki setup.


    LOKI_URL = "http://your-loki-address/loki/api/v1/query"
    end_time = time.time() + timeout
    while time.time() < end_time:
        params = {
            "query": f'{{app="openwhisk-openwhisk"}} |= "{activation_id}" |= "MessagingActiveAck"'
        }
        response = requests.get(LOKI_URL, params=params)
        result = response.json()

        # Check if any result contains MessagingActiveAck
        if result['data']['result']:
            print(f"Activation {activation_id} completed.")
            return True
        time.sleep(1)
    print(f"Timeout waiting for activation {activation_id} to complete.")
    return False


def delete_pod(namespace, pod_name):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    v1.delete_namespaced_pod(name=pod_name, namespace=namespace)
    print(f"Pod {pod_name} deleted.")


def drain_and_reclaim(namespace, pod_name, activation_id):
    label_pod_drain(namespace, pod_name)
    if wait_for_activation_completion("your-loki-query", activation_id):
        delete_pod(namespace, pod_name)
    else:
        print("Activation did not complete in time; pod not deleted.")

# Example usage:
# drain_and_reclaim(namespace="openwhisk", pod_name="wskopenwhisk-invoker-00-1967-guest-objectdetection", activation_id="2c40146f6ff5409580146f6ff59095fd")


if __name__ == "__main__":
    ranker = KubectlPodRanker(
        namespace="openwhisk",
        loki_url="http://localhost:3100/loki/api/v1/query",
        function_keywords=["guest", "objectdetection"]
    )

    top_pods = ranker.rank_pods(top_r=3)

    print("\nReady to drain these pods:")
    for pod in top_pods:
        print(f"  {pod}")