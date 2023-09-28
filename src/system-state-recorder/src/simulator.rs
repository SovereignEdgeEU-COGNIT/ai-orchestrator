use crate::monitor::Monitor;
use async_trait::async_trait;
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;
use std::sync::Mutex;

pub struct VM {
    pub vmid: String,
    pub mem: i64,
    pub cpu: i64,
}

pub struct Host {
    pub hostid: String,
    pub vms: Vec<VM>,
    pub mem: i64,
    pub cpu: i64,
}

pub struct Simulator {
    hosts: Arc<Mutex<HashMap<String, Host>>>,
}

pub trait SimulatorFactory {
    fn new(hosts: Arc<Mutex<HashMap<String, Host>>>) -> Self;
}

pub trait SimulatorHelper {
    fn add_host(&mut self, host: Host);

    fn add_host_with_vms(&mut self, host_id: String, vm_ids: Vec<String>, mem: i64, cpu: i64) {
        let vms: Vec<VM> = vm_ids
            .into_iter()
            .map(|vmid| VM { vmid, mem, cpu })
            .collect();
        let host = Host {
            hostid: host_id,
            vms,
            mem,
            cpu,
        };
        self.add_host(host);
    }

    fn add_empty_host(&mut self, host_id: String, mem: i64, cpu: i64);
}

impl SimulatorFactory for Simulator {
    fn new(hosts: Arc<Mutex<HashMap<String, Host>>>) -> Self {
        Simulator { hosts }
    }
}

impl SimulatorHelper for Simulator {
    fn add_host(&mut self, host: Host) {
        let mut hosts_guard = self.hosts.lock().unwrap();
        hosts_guard.insert(host.hostid.clone(), host);
    }
    fn add_empty_host(&mut self, host_id: String, mem: i64, cpu: i64) {
        let host = Host {
            hostid: host_id,
            vms: Vec::new(),
            mem: mem,
            cpu: cpu,
        };
        self.add_host(host);
    }
}

#[async_trait]
impl Monitor for Simulator {
    async fn get_host_vms(&self, hostid: &String) -> Result<String, Box<dyn Error>> {
        let hosts_guard = self.hosts.lock().unwrap();
        let vms_count = hosts_guard
            .get(hostid)
            .map(|host| host.vms.len())
            .ok_or_else(|| format!("Host {} not found", hostid))?;

        Ok(vms_count.to_string())
    }

    async fn get_host(&self, vmid: &String) -> Result<Option<String>, Box<dyn Error>> {
        let hosts_guard = self.hosts.lock().unwrap();
        for (hostid, host) in hosts_guard.iter() {
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
        let hosts_guard = self.hosts.lock().unwrap();
        for (hostid, host) in hosts_guard.iter() {
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
        let hosts_guard = self.hosts.lock().unwrap();
        for (host_id, _) in hosts_guard.iter() {
            result.insert(host_id.clone(), "2".to_string());
        }
        Ok(result)
    }

    async fn check_host_exists(&self, hostid: &String) -> Result<bool, Box<dyn Error>> {
        let hosts_guard = self.hosts.lock().unwrap();
        Ok(hosts_guard.contains_key(hostid))
    }

    async fn get_vms(&self) -> Result<HashMap<String, String>, Box<dyn Error>> {
        let mut vms_map = HashMap::new();

        let hosts_guard = self.hosts.lock().unwrap();
        for host in hosts_guard.values() {
            for vm in &host.vms {
                vms_map.insert(vm.vmid.clone(), "2".to_string());
            }
        }

        Ok(vms_map)
    }

    async fn get_host_total_mem(&self, hostid: &String) -> Result<String, Box<dyn Error>> {
        let hosts_guard = self.hosts.lock().unwrap();
        match hosts_guard.get(hostid) {
            Some(host) => Ok(host.mem.to_string()),
            None => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "hostid not found",
            ))),
        }
    }

    async fn get_host_usage_mem(&self, hostid: &String) -> Result<String, Box<dyn Error>> {
        let hosts_guard = self.hosts.lock().unwrap();
        match hosts_guard.get(hostid) {
            Some(host) => {
                let total_mem_used: i64 = host.vms.iter().map(|vm| vm.mem).sum();
                Ok(total_mem_used.to_string())
            }
            None => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "hostid not found",
            ))),
        }
    }

    async fn get_cpu_usage(&self, hostid: &String) -> Result<String, Box<dyn Error>> {
        let hosts_guard = self.hosts.lock().unwrap();
        match hosts_guard.get(hostid) {
            Some(host) => {
                let total_cpu_used: i64 = host.vms.iter().map(|vm| vm.cpu).sum();
                Ok(total_cpu_used.to_string())
            }
            None => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "hostid not found",
            ))),
        }
    }

    async fn get_cpu_total(&self, hostid: &String) -> Result<String, Box<dyn Error>> {
        let hosts_guard = self.hosts.lock().unwrap();
        match hosts_guard.get(hostid) {
            Some(host) => Ok(host.cpu.to_string()),
            None => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "hostid not found",
            ))),
        }
    }
}
