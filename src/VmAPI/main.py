from fastapi import FastAPI
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

class MonitorInfo(BaseModel):
    vm_path: str
    host_config: List[List[float]]


app = FastAPI()


@app.post("/MonitorInfos/")
async def root(basic_info: MonitorInfo):
    root_path = basic_info.vm_path
    hosts_infos = basic_info.host_config
    vm_map_infos = match(root_path, hosts_infos)
    return vm_map_infos

