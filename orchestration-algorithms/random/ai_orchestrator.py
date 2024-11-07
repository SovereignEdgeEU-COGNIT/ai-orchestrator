from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
import json

app = FastAPI()

class VM(BaseModel):
    ID: int
    HOST_IDS: Optional[List[int]]
    HOST_ID: Optional[int] = None

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

        # Randomize the host based on the VM ID
        host_id = hosts[int(vm_data['ID']) % len(hosts)]

        vms.append({'ID': vm_data['ID'], 'HOST_ID': host_id})

    response = {"VMS": vms}

    print("----------------- response:")
    print(json.dumps(response, indent=4))

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4567)

