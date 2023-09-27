use crate::monitor::Monitor;
use async_trait::async_trait;
use dotenv::dotenv;
use reqwest;
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::error::Error;

pub fn get_prometheus_url() -> String {
    dotenv().ok();
    env::var("PROMETHEUS_URL").unwrap_or_else(|_| "http://localhost:9090".to_string())
}

pub async fn fetch_metric_from_prometheus(
    url: &str,
    metric_name: &str,
) -> Result<Value, Box<dyn Error>> {
    let resp = reqwest::get(&format!("{}/api/v1/query?query={}", url, metric_name))
        .await?
        .json::<Value>()
        .await?;

    Ok(resp)
}

pub fn parse_vm_host_id_json(data: &Value) -> Option<String> {
    data.get("data")?
        .get("result")?
        .as_array()?
        .get(0)?
        .get("value")?
        .as_array()?
        .get(1)?
        .as_str()
        .map(String::from)
}

pub fn extract_value_from_json(data: &Value) -> Option<String> {
    data.get("data")?
        .get("result")?
        .as_array()?
        .get(0)?
        .get("value")?
        .as_array()?
        .get(1)?
        .as_str()
        .map(String::from)
}

pub fn parse_hostdata_to_dictionary(data: &Value) -> HashMap<String, String> {
    let mut result = HashMap::new();

    if let Some(result_array) = data["data"]["result"].as_array() {
        for item in result_array {
            if let Some(host_id) = item["metric"]["one_host_id"].as_str() {
                if let Some(value) = item["value"]
                    .as_array()
                    .and_then(|arr| arr.get(1))
                    .and_then(Value::as_str)
                {
                    result.insert(host_id.to_string(), value.to_string());
                }
            }
        }
    }
    result
}

fn parse_vmdata_to_dictionary(data: &Value) -> HashMap<String, String> {
    let mut result = HashMap::new();

    if let Some(result_array) = data["data"]["result"].as_array() {
        for item in result_array {
            if let Some(vm_id) = item["metric"]["one_vm_id"].as_str() {
                if let Some(value) = item["value"]
                    .as_array()
                    .and_then(|arr| arr.get(1))
                    .and_then(Value::as_str)
                {
                    result.insert(vm_id.to_string(), value.to_string());
                }
            }
        }
    }
    result
}

pub struct PrometheusMonitor;

#[async_trait]
pub trait MonitorUtils: Monitor {}

impl<T: Monitor + ?Sized> MonitorUtils for T {}

#[async_trait]
impl Monitor for PrometheusMonitor {
    async fn generate_host_vm_map(
        &self,
        vm_ids: &HashMap<String, String>,
    ) -> Result<HashMap<String, Vec<String>>, Box<dyn Error>> {
        let mut host_to_vms_map = HashMap::new();

        for vm_id in vm_ids.keys() {
            let vm_host_id_opt = self.get_host(vm_id).await?;
            if let Some(vm_host_id) = vm_host_id_opt {
                host_to_vms_map
                    .entry(vm_host_id)
                    .or_insert_with(Vec::new)
                    .push(vm_id.clone());
            }
        }

        Ok(host_to_vms_map)
    }
    async fn get_host(&self, vm_id: &String) -> Result<Option<String>, Box<dyn std::error::Error>> {
        let metric: &str = &format!("opennebula_vm_host_id{{one_vm_id=\"{}\"}}", vm_id);
        let response = fetch_metric_from_prometheus(&get_prometheus_url(), metric).await?;
        Ok(parse_vm_host_id_json(&response))
    }

    async fn get_host_vms(&self, hostid: &String) -> Result<String, Box<dyn std::error::Error>> {
        let metric: &str = &format!("opennebula_host_vms{{one_host_id=\"{}\"}}", hostid);
        let response = fetch_metric_from_prometheus(&get_prometheus_url(), metric).await?;
        println!("{:?}", response);

        match extract_value_from_json(&response) {
            Some(value) => Ok(value),
            None => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "failed to extract value from JSON",
            ))),
        }
    }

    async fn get_host_total_mem(
        &self,
        hostid: &String,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let metric: &str = &format!(
            "opennebula_host_mem_total_bytes{{one_host_id=\"{}\"}}",
            hostid
        );
        let response = fetch_metric_from_prometheus(&get_prometheus_url(), metric).await?;

        match extract_value_from_json(&response) {
            Some(value) => Ok(value),
            None => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "failed to extract value from JSON",
            ))),
        }
    }

    async fn get_host_usage_mem(
        &self,
        hostid: &String,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let metric: &str = &format!(
            "opennebula_host_mem_usage_bytes{{one_host_id=\"{}\"}}",
            hostid
        );
        let response = fetch_metric_from_prometheus(&get_prometheus_url(), metric).await?;

        match extract_value_from_json(&response) {
            Some(value) => Ok(value),
            None => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "failed to extract value from JSON",
            ))),
        }
    }

    async fn get_cpu_total(&self, hostid: &String) -> Result<String, Box<dyn std::error::Error>> {
        let metric: &str = &format!(
            "opennebula_host_cpu_total_ratio{{one_host_id=\"{}\"}}",
            hostid
        );
        let response = fetch_metric_from_prometheus(&get_prometheus_url(), metric).await?;

        match extract_value_from_json(&response) {
            Some(value) => Ok(value),
            None => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "failed to extract value from JSON",
            ))),
        }
    }

    async fn get_cpu_usage(&self, hostid: &String) -> Result<String, Box<dyn std::error::Error>> {
        let metric: &str = &format!(
            "opennebula_host_cpu_usage_ratio{{one_host_id=\"{}\"}}",
            hostid
        );
        let response = fetch_metric_from_prometheus(&get_prometheus_url(), metric).await?;

        match extract_value_from_json(&response) {
            Some(value) => Ok(value),
            None => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "failed to extract value from JSON",
            ))),
        }
    }

    async fn get_state(&self, hostid: &String) -> Result<String, Box<dyn std::error::Error>> {
        let metric: &str = &format!("opennebula_host_state{{one_host_id=\"{}\"}}", hostid);
        let response = fetch_metric_from_prometheus(&get_prometheus_url(), metric).await?;

        match extract_value_from_json(&response) {
            Some(value) => Ok(value),
            None => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "failed to extract value from JSON",
            ))),
        }
    }

    async fn get_vms_for_host(
        &self,
        host_id: &String,
        host_ids: &HashMap<String, String>,
        host_to_vms_map: &HashMap<String, Vec<String>>,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        if !host_ids.contains_key(host_id) {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "host_id not found",
            )));
        }

        let vms_for_given_host = host_to_vms_map.get(host_id).cloned().unwrap_or_default();
        Ok(vms_for_given_host)
    }

    async fn get_hosts(&self) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
        let metric: &str = "opennebula_host_state";
        let response = fetch_metric_from_prometheus(&get_prometheus_url(), metric).await?;
        Ok(parse_hostdata_to_dictionary(&response))
    }

    async fn check_host_exists(&self, hostid: &String) -> Result<bool, Box<dyn std::error::Error>> {
        let hosts = self.get_hosts().await?;
        Ok(hosts.contains_key(hostid))
    }

    async fn get_vms(&self) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
        let metric: &str = "opennebula_vm_state";
        let response = fetch_metric_from_prometheus(&get_prometheus_url(), metric).await?;
        Ok(parse_vmdata_to_dictionary(&response))
    }
}
