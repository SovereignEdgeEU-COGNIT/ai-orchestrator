from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
import httpx

app = FastAPI()

class Capacity(BaseModel):
    CPU: int
    DISK_SIZE: int
    MEMORY: int

class VM(BaseModel):
    ID: int
    HOST_IDS: Optional[List[int]]
    HOST_ID: Optional[int] = None
    CAPACITY: Capacity


async def calculate_host_priority(host_details: dict) -> (bool, float):
    memory_load = host_details["usage_mem_bytes"] / host_details["total_mem_bytes"]
    cpu_load = host_details["cpu_usage"] / host_details["cpu_total"]
    
    # Simple average of CPU and Memory load.
    load = (memory_load + cpu_load) / 2
    
    # Return renewable energy status and load as a tuple
    # False comes before True in sorting, so invert the boolean
    return (not host_details["state"]["renewable_energy"], load)

async def calculate_host_load(host_details: dict) -> float:
    memory_load = host_details["usage_mem_bytes"] / host_details["total_mem_bytes"]
    cpu_load = host_details["cpu_usage"] / host_details["cpu_total"]

    # Simple average of CPU and Memory load.
    load = (memory_load + cpu_load) / 2
    
    # If the host uses renewable energy, decrease its effective load by a factor 
    # to prioritize it in the sorting. You can adjust this factor.
    if host_details["state"]["renewable_energy"]:
        load *= 0.5

    return load

async def fetch_host_details(host_id: int) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://localhost:8000/hosts/{host_id}")

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch host details")

    return response.json()

@app.post("/")
async def root(request: Request):
    data = await request.json()

    print("----------------- request:")
    print(json.dumps(data, indent=4))

    vms = []
    for vm_data in data.get('VMS', []):
        hosts = vm_data.get('HOST_IDS', [])

        if not hosts:
            continue

        # Fetch host details and calculate load for sorting
        host_loads = []
        for host_id in hosts:
            host_details = await fetch_host_details(host_id)
            load = await calculate_host_load(host_details)
            host_loads.append((host_id, load, host_details))


        host_priorities = []
        for host_id in hosts:
            host_details = await fetch_host_details(host_id)
            priority = await calculate_host_priority(host_details)
            host_priorities.append((host_id, priority, host_details))

        # Sort hosts by renewable energy first, then by load
        sorted_hosts = sorted(host_priorities, key=lambda x: x[1])

        selected_host_id = None
        for host_id, _, host_details in sorted_hosts:
            # Convert total_mem_bytes and usage_mem_bytes to same unit as VM (bytes -> bytes)
            host_available_memory = host_details["total_mem_bytes"] - host_details["usage_mem_bytes"]
            host_available_cpu = host_details["cpu_total"] - host_details["cpu_usage"]

            vm_capacity = vm_data["CAPACITY"]
            if (host_available_cpu >= vm_capacity["CPU"] and
                    host_available_memory >= vm_capacity["MEMORY"]):
                selected_host_id = host_id
                break

        if selected_host_id:
            vms.append({'ID': vm_data['ID'], 'HOST_ID': selected_host_id})

    response = {"VMS": vms}

    print("----------------- response:")
    print(json.dumps(response, indent=4))

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4567)
