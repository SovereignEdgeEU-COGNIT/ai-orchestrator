use crate::monitor::Monitor;
use async_trait::async_trait;
use std::collections::HashMap;
use std::error::Error;

pub struct VM {
    vmid: String,
}

pub struct Host {
    hostid: String,
    vms: Vec<VM>,
}

pub struct Simulator {
    hosts: HashMap<String, Host>,
}

pub trait SimulatorHelper {
    fn new() -> Self;
    fn add_host(&mut self, host: Host);

    fn add_host_with_vms(&mut self, host_id: String, vm_ids: Vec<String>) {
        let vms: Vec<VM> = vm_ids.into_iter().map(|vmid| VM { vmid }).collect();
        let host = Host {
            hostid: host_id,
            vms,
        };
        self.add_host(host);
    }
}

impl SimulatorHelper for Simulator {
    fn new() -> Self {
        Simulator {
            hosts: HashMap::new(),
        }
    }
    fn add_host(&mut self, host: Host) {
        self.hosts.insert(host.hostid.clone(), host);
    }
}

#[async_trait]
impl Monitor for Simulator {
    async fn get_host_vms(&self, hostid: &String) -> Result<String, Box<dyn Error>> {
        let vms_count = self
            .hosts
            .get(hostid)
            .map(|host| host.vms.len())
            .ok_or_else(|| format!("Host {} not found", hostid))?;

        Ok(vms_count.to_string())
    }

    async fn get_cpu_usage(&self, _hostid: &String) -> Result<String, Box<dyn Error>> {
        Ok("200".to_string())
    }

    async fn get_host(&self, vmid: &String) -> Result<Option<String>, Box<dyn Error>> {
        for (hostid, host) in &self.hosts {
            if host.vms.iter().any(|vm| *vm.vmid == *vmid) {
                return Ok(Some(hostid.clone()));
            }
        }
        Ok(None)
    }

    async fn get_state(&self, _hostid: &String) -> Result<String, Box<dyn Error>> {
        Ok("2".to_string())
    }

    async fn generate_host_vm_map(
        &self,
        _vm_ids: &HashMap<String, String>,
    ) -> Result<HashMap<String, Vec<String>>, Box<dyn Error>> {
        let mut map = HashMap::new();

        for (hostid, host) in &self.hosts {
            let vm_ids_for_host: Vec<String> = host.vms.iter().map(|vm| vm.vmid.clone()).collect();
            map.insert(hostid.clone(), vm_ids_for_host);
        }
        Ok(map)
    }

    async fn get_vms_for_host(
        &self,
        host_id: &String,
        _host_ids: &HashMap<String, String>, // It seems you're not using this, consider removing
        host_to_vms_map: &HashMap<String, Vec<String>>,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        match host_to_vms_map.get(host_id) {
            Some(vms) => Ok(vms.clone()),
            None => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("No VMs found for host ID: {}", host_id),
            ))),
        }
    }

    async fn get_hosts(&self) -> Result<HashMap<String, String>, Box<dyn Error>> {
        let mut result = HashMap::new();
        for (host_id, _host) in &self.hosts {
            result.insert(host_id.clone(), "2".to_string());
        }
        Ok(result)
    }

    async fn check_host_exists(&self, hostid: &String) -> Result<bool, Box<dyn Error>> {
        Ok(self.hosts.contains_key(hostid))
    }

    async fn get_vms(&self) -> Result<HashMap<String, String>, Box<dyn Error>> {
        let mut vms_map = HashMap::new();

        for host in self.hosts.values() {
            for vm in &host.vms {
                vms_map.insert(vm.vmid.clone(), "2".to_string());
            }
        }

        Ok(vms_map)
    }

    async fn get_host_total_mem(&self, _hostid: &String) -> Result<String, Box<dyn Error>> {
        Ok("8330006528".to_string())
    }

    async fn get_host_usage_mem(&self, _hostid: &String) -> Result<String, Box<dyn Error>> {
        Ok("1610612736".to_string())
    }

    async fn get_cpu_total(&self, _hostid: &String) -> Result<String, Box<dyn Error>> {
        Ok("1600".to_string())
    }
}
