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
#import xmltodict
import xml.etree.ElementTree as ET
import os
import matplotlib

result_path = r"/home/zhou/project/VM_Orchestration-main/out"
json_temp = r"/home/zhou/project/VM_Orchestration-main/vm_placement_temp.json"

def extract_info(json_file):
    tree = ET.parse(json_file)
    root = tree.getroot()
    TIMESTAMP = root.find("STIME").text

    vm_index = root.find('ID').text

    for subnode in root.findall('MONITORING'):
        #print("Find monitoring")
        CPU = subnode.find('CPU').text
        MEMORY = subnode.find('MEMORY').text
        #TIMESTAMP = subnode.find('TIMESTAMP').text
        #print("CPU: ", CPU)
        #print("MEMORY: ", MEMORY)
        #print("TIMESTAMP: ", TIMESTAMP)
    return int(vm_index), float(CPU), float(MEMORY)


def ffd(vm_infos, hosts_infos):
    vm_map = {}
    for i in range(len(vm_infos)):
        assigned = False
        for j in range(len(hosts_infos)):
            if vm_infos[i][1] <= hosts_infos[j][0] and vm_infos[i][2] <= hosts_infos[j][1]:
                vm_map[vm_infos[i][0]] = j
                hosts_infos[j][0] = hosts_infos[j][0] - vm_infos[i][1]
                hosts_infos[j][1] = hosts_infos[j][1] - vm_infos[i][2]
                #print("assign vm: ", i)
                print("{}th host all {} hosts, vm {} allocate cpu {}, left {}".format(
                        j, len(hosts_infos), i, vm_infos[i][1], hosts_infos[j][0]))
                print("                       vm {} allocate mem {}, left {}".format(
                        i, vm_infos[i][2], hosts_infos[j][1]))
                assigned = True
                break
            else:
                continue
            
        if assigned == False:
            print("Build New host")
            new_host = [16400.0, 395890132.0]
            hosts_infos.append(new_host)
            vm_map[vm_infos[i][0]] = j+1
            hosts_infos[-1][0] = hosts_infos[-1][0] - vm_infos[i][1]
            hosts_infos[-1][1] = hosts_infos[-1][1] - vm_infos[i][2]
    
    print("Totally number of hosts: ", len(hosts_infos))
    
                
    return vm_map, hosts_infos
          
def gen_json(vm_ind, host_ind):
    
    f_temp = open(json_temp, 'r', encoding='utf-8')
    data = json.load(f_temp)
    print(data["properties"]["PLACEMENT"]["properties"]["VM"]["items"]["properties"]["VM_ID"])
    print(data["properties"]["PLACEMENT"]["properties"]["VM"]["items"]["properties"]["HOST_ID"])
    data["properties"]["PLACEMENT"]["properties"]["VM"]["items"]["properties"]["VM_ID"] = vm_ind
    data["properties"]["PLACEMENT"]["properties"]["VM"]["items"]["properties"]["HOST_ID"] = host_ind
    print("After Correction")
    print(data["properties"]["PLACEMENT"]["properties"]["VM"]["items"]["properties"]["VM_ID"])
    print(data["properties"]["PLACEMENT"]["properties"]["VM"]["items"]["properties"]["HOST_ID"])
    out_path = os.path.join(result_path, "vm{}_schedule.json".format(vm_ind))
    out_file = open(out_path, "w") 
    json.dump(data, out_file, indent=2) 
    
    out_file.close()
    f_temp.close()

    return data


def match(root_path, hosts_infos):
    collection = []
    vm_infos = {}
    result_jsons = []
    for file in os.listdir(root_path):
        #print("Start analysis")
        info = extract_info(os.path.join(root_path, file))
        #print(info)
        collection.append(info)
    #sort the data
    vm_infos = sorted(collection, key=lambda x:x[1], reverse=True)
    
    print("Totally number of Vms: ", len(vm_infos))
    cpu_requirements = 0
    mem_requirements = 0

    for item in collection:
        cpu_requirements += item[1]
        mem_requirements += item[2]

    vm_map, host_infos = ffd(vm_infos, hosts_infos)
    print(vm_map, type(vm_map))

    for vm_ind in vm_map:
        #print(vm_host, type(vm_host))
        host_ind = vm_map[vm_ind]
        json_data = gen_json(vm_ind, host_ind)
        result_jsons.append(json_data)

    #return result_jsons
    return vm_map


if __name__ == '__main__':
        
# =============================================================================
#     host0_cpu = 16400
#     host0_mem = 395890132
#     
#     host1_cpu = 12800
#     host1_mem = 792421016
# =============================================================================
    

    root_path = r"/home/zhou/project/VM_Orchestration-main/OpenNebula-datasets/OpenNebula-datasets/vmindividualshow"
    hosts_infos = [[16400.0, 395890132.0], [12800.0, 792421016.0]]

    
    vm_map_infos = match(root_path, hosts_infos)

   













