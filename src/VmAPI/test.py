import time, os
import json
import ast

import requests

'''
{
  "VMpool": [{'$schema': 'http://json-schema.org/2020‐12/schema#', "type": "object", "properties": {"SERVERLESS_RUNTIME": {"type": "object", "properties": {"VM": {"type": "array", "items": {"type": "object", "properties": {"VM_ID": "145148", "SERVICE_ID": {"type": "integer"}, "STATUS": {"type": "string"}, "HOSTS": {"type": "array", "properties": {"HOST_ID": {"type": "integer"}}}, "REQUIREMENTS": {"type": "object", "properties": {"CPUS": "102.75", "FLOPS": {"type": "integer"}, "MEMORY": "2150612", "DISK_SIZE": {"type": "integer"}, "IOPS": {"type": "integer"}, "LATENCY": {"type": "integer"}, "BANDWIDTH": {"type": "integer"}, "ENERGY": {"type": "integer"}}}}}}}}}}],
  "Host_config": [[16400.0, 395890132.0], [12800.0, 792421016.0]]
}
'''

input_path = "/home/zhou/repo/VM_Orchestration/fake_input"

def test():
    url = 'http://127.0.0.1:5200/VM_placement'
    print('Test url:', url)

    ##generate input list(dict)
    
    vms_infos = []
    for json_input in os.listdir(input_path):
        print(json_input)
        f_temp = open(os.path.join(input_path, json_input), 'r')
        data = json.load(f_temp)
        #print("Before dumps: ", type(data))
        #data = json.dumps(data)

        print(data, type(data))
        vms_infos.append(data)
        f_temp.close()
        #break
    
    
    
    
    params = {"VMpool": vms_infos, "Host_config": [[16400.0, 395890132.0], [12800.0, 792421016.0]]}
    params = json.dumps(params)
    #params = {
    #          "VMpool": [{"$schema": "http://json-schema.org/2020‐12/schema#", "type": "object", "properties": {"SERVERLESS_RUNTIME": {"type": "object", "properties": {"VM": {"type": "array", "items": {"type": "object", "properties": {"VM_ID": "145148", "SERVICE_ID": {"type": "integer"}, "STATUS": {"type": "string"}, "HOSTS": {"type": "array", "properties": {"HOST_ID": {"type": "integer"}}}, "REQUIREMENTS": {"type": "object", "properties": {"CPUS": "102.75", "FLOPS": {"type": "integer"}, "MEMORY": "2150612", "DISK_SIZE": {"type": "integer"}, "IOPS": {"type": "integer"}, "LATENCY": {"type": "integer"}, "BANDWIDTH": {"type": "integer"}, "ENERGY": {"type": "integer"}}}}}}}}}}],
    #          "Host_config": [[16400.0, 395890132.0], [12800.0, 792421016.0]]
    #         }
    
    #test_para = {"VMpool": [{"name": "zz"}], "Host_config": [[16400.0, 395890132.0], [12800.0, 792421016.0]]}

    #params = json.dumps(params)

    #print(type(params))
    #print(params)
    output = requests.post(url, params)
    print('返回:', output.text)


test()
