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
    active_vms = {} 
    vmid = 1

    while True:
        current_time = time.time()
        print("current_time:", current_time)
        keys_to_delete = []
        for id, timestamp in active_vms.items():
            print("id:", id)
            print("timestamp:", timestamp)
            if current_time > timestamp:
                keys_to_delete.append(id)
       
        for id in keys_to_delete:
            print("deleting vm with id=", id)
            del active_vms[id]
            delete_vm(id)

        print("len(active_vms):", len(active_vms))
        if len(active_vms) < 20:
            place_vm(vmid, 1073741824, 2000)
        
            lifespan = vm_lifespan()
            expire_time = current_time + lifespan
            active_vms[vmid] = expire_time
        
            print(f"vm {vmid} lifespan: {lifespan:.2f} seconds")
            vmid += 1

        time.sleep(1)

def main():
    for i in range(0, 9):
        add_host(i, 8073741824, 16000)
   
    manage_vms()

if __name__ == "__main__":
    main()

