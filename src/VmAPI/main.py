from fastapi import FastAPI
import uvicorn
from VM_info import match
from pydantic import BaseModel
from typing import Optional, Union, List

'''
{
  "vm_path": "/home/zhou/project/VM_Orchestration-main/OpenNebula-datasets/OpenNebula-datasets/vmindividualshow",
  "host_config": [
    [16400.0, 395890132.0], [12800.0, 792421016.0]
  ]
}
'''

class VMsConfig(BaseModel):
    VMpool: List[dict]
    Host_config: List[List[float]]
    


app = FastAPI()


@app.post("/VM_placement/")
async def VM_placement(Vms_info: VMsConfig):
    vms_infos = Vms_info.VMpool
    hosts_infos = Vms_info.Host_config
    vm_map_infos = match(vms_infos, hosts_infos)
    return vm_map_infos

if __name__ == '__main__':
    uvicorn.run(app=app, host="127.0.0.1", port=5200)