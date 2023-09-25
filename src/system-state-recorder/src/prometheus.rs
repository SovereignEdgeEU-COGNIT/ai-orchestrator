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
            if let Some(host_id) = item["metric"]["one_vm_id"].as_str() {
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

pub async fn get_host(vm_id: &String) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let metric: &str = &format!("opennebula_vm_host_id{{one_vm_id=\"{}\"}}", vm_id);
    let response = fetch_metric_from_prometheus(PROMETHEUS_URL, metric).await?;
    Ok(parse_vm_host_id_json(&response))
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

// #[tokio::main]
// async fn main() -> Result<(), Box<dyn Error>> {
//     println!("printing all known hosts:");
//     let host_ids = get_hosts().await?;
//     for host_id in host_ids.keys() {
//         println!(" host_id={}", host_id);
//     }

//     println!("printing all vms:");
//     let vms_ids = get_vms().await?;
//     for vm_id in vms_ids.keys() {
//         println!(" vm_id={}", vm_id);
//     }

//     println!("mapping vm to hosts:");
//     for vm_id in vms_ids.keys() {
//         let host_id = get_host(vm_id).await?;
//         match host_id {
//             Some(host_id) => println!("vm_id={} is running on host_id={}", vm_id, host_id),
//             None => println!("Failed to retrieve host ID"),
//         }
//     }

//     Ok(())
// }
