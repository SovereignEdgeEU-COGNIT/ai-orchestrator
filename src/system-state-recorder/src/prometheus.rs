use reqwest;
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;

const PROMETHEUS_URL: &str = "http://localhost:9090";

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

pub async fn get_host(vm_id: &String) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let metric: &str = &format!("opennebula_vm_host_id{{one_vm_id=\"{}\"}}", vm_id);
    let response = fetch_metric_from_prometheus(PROMETHEUS_URL, metric).await?;
    Ok(parse_vm_host_id_json(&response))
}

pub async fn generate_host_vm_map(
    vm_ids: &HashMap<String, String>,
) -> Result<HashMap<String, Vec<String>>, Box<dyn Error>> {
    let mut host_to_vms_map = HashMap::new();

    for vm_id in vm_ids.keys() {
        let vm_host_id_opt = get_host(vm_id).await?;
        if let Some(vm_host_id) = vm_host_id_opt {
            host_to_vms_map
                .entry(vm_host_id)
                .or_insert_with(Vec::new)
                .push(vm_id.clone());
        }
    }

    Ok(host_to_vms_map)
}

pub async fn get_vms_for_host(
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

pub async fn get_vms_hosts(
    host_ids: &HashMap<String, String>,
    vm_ids: &HashMap<String, String>,
) -> Result<HashMap<String, Vec<String>>, Box<dyn std::error::Error>> {
    let host_to_vms_map = generate_host_vm_map(vm_ids).await?;
    let filtered_map: HashMap<String, Vec<String>> = host_to_vms_map
        .into_iter()
        .filter(|(host_id, _)| host_ids.contains_key(host_id))
        .collect();

    Ok(filtered_map)
}

pub async fn get_hosts() -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    let metric: &str = "opennebula_host_state";
    let response = fetch_metric_from_prometheus(PROMETHEUS_URL, metric).await?;
    Ok(parse_hostdata_to_dictionary(&response))
}

pub async fn get_vms() -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    let metric: &str = "opennebula_vm_state";
    let response = fetch_metric_from_prometheus(PROMETHEUS_URL, metric).await?;
    //println!("{:#?}", response);
    Ok(parse_vmdata_to_dictionary(&response))
}
