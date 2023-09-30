import requests
import numpy as np
import threading
import time
from collections import deque

BASE_URL = "http://localhost:8000"
HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded"
}

def send_request(method, endpoint, params=None):
    url = f"{BASE_URL}/{endpoint}"
    response = getattr(requests, method)(url, headers=HEADERS, params=params)
    print(f"{method.upper()} {endpoint} - Params {params} - Status Code: {response.status_code}")

def place_vm(vmid, mem, cpu):
    params = {
        "vmid": vmid,
        "mem": mem,
        "cpu": cpu
    }
    send_request("post", "placevm", params)

def add_host(hostid, mem, cpu):
    params = {
        "hostid": hostid,
        "mem": mem,
        "cpu": cpu
    }
    send_request("post", "addhost", params)

def vm_lifespan(alpha=2, beta=5, scale=100):
    return np.random.beta(alpha, beta) * scale

def delete_vm(vmid):
    endpoint = f"vms/{vmid}"
    send_request("delete", endpoint)

def manage_vms():
    active_vms = deque()
    vmid = 1

    while True:
        # Delete VMs whose lifespan has expired
        current_time = time.time()
        while active_vms and active_vms[0][1] <= current_time:
            _, expiring_vmid = active_vms.popleft()
            delete_vm(expiring_vmid)
        
        # Spawn new VMs if less than 20 are active
        #while len(active_vms) < 20:
        place_vm(vmid, 1073741824, 2000)
        
        lifespan = vm_lifespan()
        expire_time = current_time + lifespan
        active_vms.append((expire_time, vmid))
        
        print(f"VM {vmid} lifespan: {lifespan:.2f} seconds")
        vmid += 1

        time.sleep(1)  # Sleep for a while before checking again

def main():
    for i in range(0, 9):
        add_host(i, 8073741824, 16000)
   
    manage_vms()
    # for i in range(0, 20):
    #     cpu_val = 2000 if i < 11 else 5000
    #     place_vm(i, 1073741824, cpu_val)

if __name__ == "__main__":
    main()

