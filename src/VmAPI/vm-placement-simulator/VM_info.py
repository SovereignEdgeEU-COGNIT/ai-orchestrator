# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 08:45:49 2023

@author: nikcy zhou
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:14:22 2023

@author: nikcy zhou
"""

import json
import os
import requests
import random


def analysis_vmjson(json_data):
    
    vm_infos = []
    hosts_set = []
    for i in range(len(json_data['VMS'])):
        vm_id = json_data['VMS'][i]['ID']
        #print("vm id: ", vm_id)
        vm_cpu = json_data['VMS'][i]['CAPACITY']['CPU']
        vm_mem = json_data['VMS'][i]['CAPACITY']['MEMORY']
        aval_hosts = json_data['VMS'][i]['HOST_IDS']
        #print("vm id: ", vm_id)
        #print("vm_cpu: ", vm_cpu)
        #print("vm_mem: ", vm_mem)
        #print("available hosts: ", aval_hosts)
        vm_info = [int(vm_id), int(vm_cpu), int(vm_mem)]
        vm_infos.append(vm_info)
        
        for item in aval_hosts:
            if item not in hosts_set:
                hosts_set.append(item)

    #collect configuration of hosts
    host_info_set = []
    print("----------------- Request available hosts resources:")
    for host_id in hosts_set:
        #raw_data = os.popen("curl http://localhost:8000/hosts/{}".format(int(host_id))).read()
        url = "http://192.168.113.128:8000/hosts/{}".format(int(host_id))
        raw_data = requests.get(url)
        data = raw_data.json()
        #print("host data: ", data)
        renewable = data["state"]["renewable_energy"]
        aval_mem = int(data["total_mem_bytes"]) - int(data["usage_mem_bytes"])
        aval_cpu = float(data["cpu_total"]) - int(data["cpu_usage"])
        host_info = [host_id, aval_cpu, aval_mem, renewable]
        host_info_set.append(host_info)
    
    #print(vm_infos)
    #print(host_info_set)
    return vm_infos, host_info_set


def Green_ffd(vm_infos, host_info_set):
    gHost_set = []
    ngHost_set = []
    for host_info in host_info_set:
        if host_info[-1] == False:
            ngHost_set.append(host_info)
        else:
            gHost_set.append(host_info)
    #print("Green: ", gHost_set)
    vm_map = []
    for i in range(len(vm_infos)):
        ##assign to green hosts
        G_assigned = False
        for j in range(len(gHost_set)):
            if vm_infos[i][1] <= gHost_set[j][1] and vm_infos[i][2] <= gHost_set[j][2]:
                vm_map.append([vm_infos[i][0], gHost_set[j][0]])
                gHost_set[j][1] = gHost_set[j][1] - vm_infos[i][1]
                gHost_set[j][2] = gHost_set[j][2] - vm_infos[i][2]
                G_assigned = True
                break
            else:
                continue
                
        ##assign to non green hosts
        if G_assigned == False:
            NG_assigned = False
            for j in range(len(ngHost_set)):
                if vm_infos[i][1] <= ngHost_set[j][1] and vm_infos[i][2] <= ngHost_set[j][2]:
                    vm_map.append([vm_infos[i][0], ngHost_set[j][0]])
                    ngHost_set[j][1] = ngHost_set[j][1] - vm_infos[i][1]
                    ngHost_set[j][2] = ngHost_set[j][2] - vm_infos[i][2]
                    NG_assigned = True
                    break
                else:
                    continue
        ##If the reqiurements exceeds all available resources on hosts, randomly assign a host.
        if G_assigned == False and NG_assigned == False:
            #random_host_id = random.randint(0, len(host_info_set)-1)
            vm_map.append([vm_infos[i][0], None])

    return vm_map



def match(json_data):

    json_data = {"VMS": json_data}
    vm_infos, host_info_set = analysis_vmjson(json_data)
    
    #sort the data
    sorted_vm_infos = sorted(vm_infos, key=lambda x:x[1], reverse=True)
    
    #print("Totally number of Vms: ", len(vm_infos))
    vm_map = Green_ffd(sorted_vm_infos, host_info_set)
    #print(vm_map, type(vm_map))

    pair_list = []
    for ele in vm_map:
        pair = {'ID': ele[0], 'HOST_ID': ele[1]}
        pair_list.append(pair)
    vm_map_dict = {"VMS": pair_list}
    return vm_map_dict


if __name__ == '__main__':
        
    input_path = "/home/zhou/project/vm_placcement_sim/vm_input.json"
    vm_json = open(input_path, 'r')
    json_data = json.load(vm_json)  
    vm_map_infos = match(json_data)
    print(vm_map_infos)

   













