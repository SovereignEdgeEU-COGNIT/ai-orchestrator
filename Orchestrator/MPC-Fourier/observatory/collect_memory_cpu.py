import subprocess
import csv
import time
from datetime import datetime

NAMESPACE = "openwhisk"
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FILE = f"memory_cpu_log_{timestamp_str}.csv"

SAMPLE_INTERVAL = 2  # seconds
RUNTIME_KEYWORDS = ["guest", "objectdetection"]  # adjust as needed

def parse_kubectl_top(output):
    lines = output.strip().split('\n')[1:]  # skip header
    total_cpu = 0
    total_mem = 0
    count = 0
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            pod_name = parts[0]
            if any(key in pod_name for key in RUNTIME_KEYWORDS):
                cpu = int(parts[1].replace('m', ''))
                mem = int(parts[2].replace('Mi', ''))
                total_cpu += cpu
                total_mem += mem
                count += 1
    return count, total_cpu, total_mem

def log_to_csv(timestamp, count, total_cpu, total_mem):
    file_exists = False
    try:
        with open(OUTPUT_FILE, 'r'):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(OUTPUT_FILE, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['timestamp', 'number_function_runtime', 'total_cpu_millicores', 'total_memory_MiB'])
        writer.writerow([timestamp, count, total_cpu, total_mem])

def main():
    MAX_RUNTIME = 6 * 60 * 60  # 6 hours in seconds
    start_time = time.time()
    while time.time() - start_time < MAX_RUNTIME:
        timestamp = datetime.utcnow().isoformat()
        try:
            result = subprocess.run(
                ["kubectl", "top", "pods", "-n", NAMESPACE],
                capture_output=True, text=True, check=True
            )
            count, total_cpu, total_mem = parse_kubectl_top(result.stdout)
            log_to_csv(timestamp, count, total_cpu, total_mem)
            print(f"[{timestamp}] Runtimes={count}, CPU={total_cpu}m, MEM={total_mem}MiB")
        except subprocess.CalledProcessError as e:
            print(f"Error fetching pod usage: {e}")
        time.sleep(SAMPLE_INTERVAL)

if __name__ == "__main__":
    main()