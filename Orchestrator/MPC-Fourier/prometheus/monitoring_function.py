import requests
import subprocess
import time
from datetime import datetime

class KubectlPodRanker:
    def __init__(self, namespace="openwhisk", loki_url="http://localhost:3100/loki/api/v1/query", function_keywords=["guest", "objectdetection"]):
        self.namespace = namespace
        self.loki_url = loki_url
        self.function_keywords = function_keywords

    def query_loki(self, logql_query):
        response = requests.get(self.loki_url, params={"query": logql_query})
        response.raise_for_status()
        return response.json()

    def get_pod_list_from_kubectl(self):
        try:
            result = subprocess.run(
                ["kubectl", "top", "pods", "-n", self.namespace],
                capture_output=True, text=True, check=True
            )
            return self.parse_kubectl_top(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running kubectl top: {e}")
            return []

    def parse_kubectl_top(self, output):
        lines = output.strip().split('\n')[1:]  # skip header
        pod_infos = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                pod_name = parts[0]
                if any(key in pod_name for key in self.function_keywords):
                    cpu_m = int(parts[1].replace('m', ''))  # CPU in millicores
                    mem_mib = int(parts[2].replace('Mi', ''))  # Memory in MiB
                    pod_infos.append({
                        'pod_name': pod_name,
                        'cpu_millicores': cpu_m,
                        'memory_mib': mem_mib
                    })
        return pod_infos

    def get_idle_seconds(self, pod_name):
        query = f'{{app="openwhisk"}} |= "sending arguments to ContainerId({pod_name})"'
        try:
            result = self.query_loki(query)
            if 'data' in result and 'result' in result['data'] and len(result['data']['result']) > 0:
                latest_timestamp_ns = max(int(value[0]) for stream in result['data']['result'] for value in stream['values'])
                latest_timestamp_s = latest_timestamp_ns / 1e9
                now = time.time()
                return now - latest_timestamp_s
            else:
                return 999999  # No activation found = very idle
        except Exception as e:
            print(f"Error fetching idle time for {pod_name}: {e}")
            return 999999

    def compute_score(self, memory_mib, cpu_millicores, idle_seconds):
        # Lower score = more idle
        return (memory_mib / 500) + (cpu_millicores / 1000) - (idle_seconds / 600)

    def rank_pods(self, top_r=3, already_deleted_pods= None):
        if already_deleted_pods is None:
            already_deleted_pods = set()
        pod_infos = self.get_pod_list_from_kubectl()
        ranked_pods = []

        for pod in pod_infos:
            if pod['pod_name'] in already_deleted_pods:
                continue  # Skip already deleted pods
            idle = self.get_idle_seconds(pod['pod_name'])
            score = self.compute_score(
                memory_mib=pod['memory_mib'],
                cpu_millicores=pod['cpu_millicores'],
                idle_seconds=idle
            )
            pod['idle_seconds'] = idle
            pod['score'] = score
            ranked_pods.append(pod)
            #print(f"Pod {pod['pod_name']} | CPU={pod['cpu_millicores']}m | MEM={pod['memory_mib']}MiB | Idle={idle:.1f}s | Score={score:.3f}")

        ranked_pods.sort(key=lambda x: x['score'])
        selected = ranked_pods[:top_r]

        #print(f"\nTop {top_r} pods to drain:")
        #for pod in selected:
        #    print(f"  {pod['pod_name']} (score={pod['score']:.3f})")

        #return [pod['pod_name'] for pod in selected]
        return selected

    def list_running_function_pods(self, v1):
        pods = v1.list_namespaced_pod(namespace=self.namespace).items
        running_pods = []
        for pod in pods:
            if pod.status.phase == "Running" and any(key in pod.metadata.name for key in self.function_keywords):
                running_pods.append(pod.metadata.name)
        return running_pods
class PrometheusClient:
    def __init__(self, ip="localhost", port=9090):
        self.base_url = f"http://{ip}:{port}/api/v1/query"

    def query(self, promql):
        try:
            response = requests.get(self.base_url, params={"query": promql}, timeout=5)
            response.raise_for_status()
            data = response.json()
            if data["status"] == "success" and data["data"]["result"]:
                return float(data["data"]["result"][0]["value"][1])
            return 0.0
        except Exception as e:
            print(f"Prometheus query failed: {e}")
            return 0.0

    #function = object-detection
    def get_request_rate(self, function, window="1m"):
        query = f'rate(openwhisk_action_activations_total{{action="{function}"}}[{window}])'

        # rate(openwhisk_action_activations_total{action="object-detection"}[1m])
        return self.query(query)

    def get_warm_container_count(self, pod_name):
        #print(f"[DEBUG] Using Prometheus base URL: {self.base_url}")

        #pod_name= "wskopenwhisk-invoker-00-.*-guest-objectdetection"
        query = f'count(kube_pod_status_phase{{namespace="openwhisk", phase="Running", pod=~"{pod_name}"}})'

        #print(f"[DEBUG] Querying Prometheus: {query}")
        result = self.query(query)
        #print(f"[DEBUG] Raw result from Prometheus: {result}")
        return self.query(query)

    #get total
    def get_cpu_usage(self, function):
        query = f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{function}.*"}}[1m]))'
        return self.query(query)

    #get total
    def get_memory_usage(self, function):
        query = f'sum(container_memory_usage_bytes{{pod=~"{function}.*"}})'
        return self.query(query)

    def get_pod_cpu_usage(self, pod_name):
        query = f'rate(container_cpu_usage_seconds_total{{pod="{pod_name}"}}[1m])'
        result = query(query)
        try:
            if 'data' in result and 'result' in result['data'] and len(result['data']['result']) > 0:
                # usually a single value
                value = float(result['data']['result'][0]['value'][1])
                return value
            else:
                return 0.0  # no data, assume no usage
        except Exception as e:
            print(f"Failed to get CPU usage for {pod_name}: {e}")
            return 0.0

    ##both warm and in progress cold start container
    #pod_regex = "wskopenwhisk-invoker-00-.*-guest-objectdetection"

    def get_total_container_count(self, pod_regex, namespace="openwhisk"):
        query = f'sum(kube_pod_status_phase{{namespace="{namespace}", phase="Running", pod=~"{pod_regex}"}})'
        value = self.query(query)
        return int(value)  # cast to int because pods are discrete

