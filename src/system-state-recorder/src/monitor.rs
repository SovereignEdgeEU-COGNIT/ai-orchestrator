use async_trait::async_trait;
use std::collections::HashMap;
use std::error::Error;

#[async_trait]
pub trait Monitor: Send + Sync {
    async fn get_host_vms(&self, hostid: &String) -> Result<String, Box<dyn std::error::Error>>;
    async fn get_cpu_usage(&self, hostid: &String) -> Result<String, Box<dyn std::error::Error>>;
    async fn get_host(&self, vm_id: &String) -> Result<Option<String>, Box<dyn std::error::Error>>;
    async fn get_state(&self, hostid: &String) -> Result<String, Box<dyn std::error::Error>>;
    async fn generate_host_vm_map(
        &self,
        vm_ids: &HashMap<String, String>,
    ) -> Result<HashMap<String, Vec<String>>, Box<dyn Error>>;
    async fn get_vms_for_host(
        &self,
        host_id: &String,
        host_ids: &HashMap<String, String>,
        host_to_vms_map: &HashMap<String, Vec<String>>,
    ) -> Result<Vec<String>, Box<dyn Error>>;
    async fn get_hosts(&self) -> Result<HashMap<String, String>, Box<dyn std::error::Error>>;
    async fn check_host_exists(&self, hostid: &String) -> Result<bool, Box<dyn std::error::Error>>;
    async fn get_vms(&self) -> Result<HashMap<String, String>, Box<dyn std::error::Error>>;
    async fn get_host_total_mem(
        &self,
        hostid: &String,
    ) -> Result<String, Box<dyn std::error::Error>>;
    async fn get_host_usage_mem(
        &self,
        hostid: &String,
    ) -> Result<String, Box<dyn std::error::Error>>;
    async fn get_cpu_total(&self, hostid: &String) -> Result<String, Box<dyn std::error::Error>>;
}
