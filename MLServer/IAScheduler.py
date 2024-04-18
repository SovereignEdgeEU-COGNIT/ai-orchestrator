import threading
from typing import Dict, List
from ClassifierInterface import ClassifierInterface
import numpy as np
import sys
from DBConnector import DBClient, Host, Vm, Metric
from SchedulerInterface import SchedulerInterface
from OnedConnector import OnedConnector


class InteferenceAwareScheduler(SchedulerInterface):
    
    def __init__(self, db: DBClient, oned: OnedConnector, classifier: ClassifierInterface) -> None:
        self.db = db
        self.oned = oned
         # 2D array of host interference-aware values
        self.ia_width = classifier.get_output_size()
        self.hosts_ia_vals = np.ndarray((0, self.ia_width))
        self.hosts_green_energy = np.ndarray((0, 1))
        self.hosts_to_index_map: Dict[str, int] = {}
        self.index_to_hosts_map: Dict[int, str] = {}
        #attributes = 6 # cpu, mem, disk_read, disk_write, net_rx, net_tx
        self.attribute_weights = np.ones(self.ia_width) / self.ia_width
        self.classifier = classifier
        self.green_energy_scalar = 1
        self.lock = threading.Lock()
     
    @staticmethod
    def get_name() -> str:
        return "InteferenceAwareScheduler"
    
    def cleanup(self): #! Rename
        self.db.close_all_connections()
        
    def set_green_energy_scalar(self, green_energy_scalar: int):
        self.green_energy_scalar = green_energy_scalar
        
    def set_classifier(self, classifier: ClassifierInterface):
        self.classifier = classifier
        self.ia_width = classifier.get_output_size()
        self.hosts_ia_vals = np.ndarray((0, self.ia_width))
        self.attribute_weights = np.ones(self.ia_width) / self.ia_width
        self.initialize()
        print("classifier loaded")
        
    #! Improve error handling at this function, bool return for if host exist or not is hard to understand
    def init_host(self, hostid: int) -> bool:
        #if host.usage_cpu == 0:
            #continue
        host = self.db.fetch_host(hostid)
        
        if host is None or len(host) == 0:
            return False
        else: 
            host = host[0]
        
        host_init_ia_vals = np.zeros(self.ia_width)
        #host_green_energy = np.random.rand(1) #! This should be the actual green energy value
        host_green_energy = self.oned.get_host_green_energy(int(host.hostid))
        #print("host_green_energy\n", host_green_energy)
        
        self.hosts_ia_vals = np.vstack([self.hosts_ia_vals, host_init_ia_vals])
        self.hosts_green_energy = np.vstack([self.hosts_green_energy, host_green_energy])
        self.hosts_to_index_map[host.hostid] = len(self.hosts_ia_vals) - 1
        self.index_to_hosts_map[len(self.hosts_ia_vals) - 1] = host.hostid
        
        vms = self.db.fetch_host_vms(int(host.hostid))
        host_index = self.hosts_to_index_map[host.hostid]
        
        for vm in vms:
            
            if vm.vmid is None or vm.vmid == '':
                continue
            vm_ia_vals = self.classifier.predict(int(vm.vmid))
            #print("vm_ia_vals\n", vm_ia_vals)
            #? The ML algorithm will give vals between 0 and 1 so mult by 100
            vm_ia_vals = np.multiply(vm_ia_vals, 100)
            #print("vm_ia_vals after mult by 100\n", vm_ia_vals)
            self.hosts_ia_vals[host_index] += vm_ia_vals
            
        return True
        #print(host.hostid)

    def initialize(self) -> None:
        hosts = self.db.fetch_hosts()
            
        self.lock.acquire()
        
        for host in hosts:
            success = self.init_host(int(host.hostid))
            
            if success == False:
                print("Failed to initialize host", host.hostid)
            else:
                print("Initialized host", host.hostid)
   
        self.lock.release()

    def predict(self, vm_id: int, host_ids: List[int]) -> int:
        
        vms = self.db.fetch_vm(vm_id)
        
        vm_ia_vals = self.classifier.predict(vm_id)
        print("vm_ia_vals\n", vm_ia_vals)
        #! The ML algorithm will give me vals between 0 and 1 so mult by 100
        vm_ia_vals = np.multiply(vm_ia_vals, 100)
        print("vm_ia_vals after mult by 100\n", vm_ia_vals)
        
        self.lock.acquire()
        
        # The indicies and rev map are confusing, need to clean this up
        host_indicies = []
        host_subset_indicies_index_map = {}
        for host_id in host_ids:
            
            if str(host_id) not in self.hosts_to_index_map:
                successful = self.init_host(host_id)
                
                if successful == False:
                    continue
                
            host_index = self.hosts_to_index_map.get(str(host_id))
            host_indicies.append(host_index)
            host_subset_indicies_index_map[len(host_indicies) - 1] = host_index
            
        if len(host_indicies) == 0:
            print(self.hosts_to_index_map)
            self.lock.release()
            return -1
                
        hosts_ia_vals_subset = self.hosts_ia_vals[np.array(host_indicies)]
        hosts_green_energy_subset = self.hosts_green_energy[np.array(host_indicies)]
        
        
        host_subset_index = self.schedule_vm(vm_ia_vals, hosts_ia_vals_subset, hosts_green_energy_subset)
        #print(host_subset_index)
        host_index = host_subset_indicies_index_map[host_subset_index]
        self.hosts_ia_vals[host_index] += vm_ia_vals
        
        if len(vms) == 1 and vms[0] is not None:
            vm = vms[0]
            if vm.hostid is not None and vm.hostid != '' and vm.hostid in self.hosts_to_index_map:
                #print("works")
                prev_host_index = self.hosts_to_index_map[vm.hostid]
                self.hosts_ia_vals[prev_host_index] -= vm_ia_vals
        
        self.lock.release()
        
        host_id = self.index_to_hosts_map.get(int(host_index))
        
        return int(host_id or -1)
    
    
        
    def schedule_vm(self, vm_ia_vals, hosts_ia_vals, hosts_green_energy) -> np.intp:
        # Adjust VM attributes based on weights, currently not used
        #weighted_vm_ia_vals = vm_ia_vals * self.attribute_weights
        #print("weighted_vm_ia_vals\n", weighted_vm_ia_vals)
        
        # Calculate weighted host attributes, currently not used
        #weighted_hosts_attributes = hosts_ia_vals * self.attribute_weights
        #print("weighted_host_attributes\n", weighted_hosts_attributes)
        
        # Calculate the average for each attribute
        weighted_host_mean_loads = np.mean(hosts_ia_vals, axis=0)
        print("weighted_host_mean_loads\n", weighted_host_mean_loads)
        
        # Calculate potential new load for each host after adding the VM, normalized by attribute importance
        inteference_adjusted_host_load = (hosts_ia_vals + vm_ia_vals) - weighted_host_mean_loads
        print("inteference_adjusted_host_load\n", inteference_adjusted_host_load)
        
        #! Need to add accounting for specific resource utilization, 
        #! e.g. don't want 1 host to have 100 cpu and 0 mem and the other to have 0 cpu and 100 mem, 
        #! currently these would have the same score
        #! The new func is in the doc
        # Calculate the load from the mean for each host
        #host_load_from_mean = host_loads - np.mean(host_loads)
        #print("host_load_from_mean\n", host_load_from_mean)

        # Apply the weight to the potential new load
        #potential_new_load += potential_new_load * (host_load_from_mean[:, np.newaxis]) * host_loading_scalar
        #print("potential_new_load after loading scaling\n", potential_new_load)
        
        # Apply lack of green energy cost
        inteference_adjusted_host_load += inteference_adjusted_host_load * (1 - hosts_green_energy) * self.green_energy_scalar
        print("inteference_adjusted_host_load after green energy cost\n", inteference_adjusted_host_load)

        # Lower scores are better. This is a simplified scoring mechanism; adjust weights as needed.
        scores = np.sum(inteference_adjusted_host_load, axis=1) #- green_energy * green_energy_weight + host_loads * host_load_weight  # Weight green energy and load heavily
        print("scores\n", scores)
        
        # Choose the host with the lowest score
        chosen_host_idx = np.argmin(scores)
        
        return chosen_host_idx
