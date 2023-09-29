from fastapi import FastAPI
import uvicorn
from VM_info import match
from pydantic import BaseModel
from typing import Optional, Union, List
import json


class VMsConfig(BaseModel):
    VMS: List[dict]
    


app = FastAPI()


@app.post("/")
async def VM_placement(Vms_info: VMsConfig):

    vms_infos = Vms_info.VMS
    #print("!!!!!!!!!!!!!!!!!!s")
    #print(vms_infos)
    print("----------------- request:")
    print(json.dumps(vms_infos, indent=4))

    vm_map_infos = match(vms_infos)
    print("----------------- response:")
    print(json.dumps(vm_map_infos, indent=4))


    return vm_map_infos

if __name__ == '__main__':
    uvicorn.run(app=app, host="0.0.0.0", port=4567)##use this setting to meet RISE infra
    
