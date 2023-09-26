extern crate rocket;
extern crate serde;

use rocket::serde::json::Json;
use serde::Deserialize;
use serde::Serialize;
use staterec::prometheus::{
    check_host_exists, generate_host_vm_map, get_cpu_total, get_cpu_usage, get_host_total_mem,
    get_host_usage_mem, get_host_vms, get_hosts, get_state, get_vms, get_vms_for_host,
};
use std::collections::HashMap;
use std::sync::RwLock;

lazy_static::lazy_static! {
    static ref GLOBAL_HOST_STATE: RwLock<HashMap<String, HostState>> = RwLock::new(HashMap::new());
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HostState {
    renewable_energy: bool,
}

#[derive(Serialize)]
pub struct Host {
    hostid: String,
    vmids: Vec<String>,
    state: HostState,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    error: String,
}

#[derive(Serialize)]
struct HostInfo {
    state: HostState,
    total_mem_bytes: i64,
    usage_mem_bytes: i64,
    cpu_total: i64,
    cpu_usage: i64,
    powerstate: i64,
    vms: String,
}

#[get("/")]
pub async fn index() -> Result<Json<Vec<Host>>, rocket::http::Status> {
    let host_ids = match get_hosts().await {
        Ok(hosts) => hosts,
        Err(_) => return Err(rocket::http::Status::InternalServerError),
    };

    let vm_ids = match get_vms().await {
        Ok(hosts) => hosts,
        Err(_) => return Err(rocket::http::Status::InternalServerError),
    };

    let mut mappings: Vec<Host> = Vec::new();

    let host_to_vms_map = match generate_host_vm_map(&vm_ids).await {
        Ok(m) => m,
        Err(_) => return Err(rocket::http::Status::InternalServerError),
    };

    for host_id in host_ids.keys() {
        let vm_ids = match get_vms_for_host(host_id, &host_ids, &host_to_vms_map).await {
            Ok(ids) => ids,
            _ => continue,
        };
        let mut host_states = match GLOBAL_HOST_STATE.write() {
            Ok(states) => states,
            Err(_) => continue,
        };

        let host_state = match host_states.entry(host_id.clone()) {
            std::collections::hash_map::Entry::Occupied(o) => o.get().clone(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let default_state = HostState {
                    renewable_energy: false,
                };
                v.insert(default_state).clone()
            }
        };

        mappings.push(Host {
            state: host_state,
            hostid: host_id.clone(),
            vmids: vm_ids,
        });
    }

    Ok(Json(mappings))
}

#[put("/set?<hostid>&<renewable>")]
pub fn set_renewable(
    hostid: String,
    renewable: bool,
) -> Result<rocket::http::Status, rocket::http::Status> {
    let new_state = HostState {
        renewable_energy: renewable,
    };

    match GLOBAL_HOST_STATE.write() {
        Ok(mut host_states) => {
            host_states.insert(hostid, new_state);
            Ok(rocket::http::Status::Ok)
        }
        Err(_) => Err(rocket::http::Status::InternalServerError),
    }
}

#[get("/hosts/<hostid>")]
pub async fn get_host_info(hostid: String) -> Result<Json<HostInfo>, Json<ErrorResponse>> {
    if !check_host_exists(&hostid)
        .await
        .map_err(|_| ErrorResponse {
            error: "internal server error".to_string(),
        })?
    {
        return Err(Json(ErrorResponse {
            error: "host not found".to_string(),
        }));
    }

    let total_mem_bytes_str = match get_host_total_mem(&hostid).await {
        Ok(mem) => mem,
        Err(_) => {
            return Err(Json(ErrorResponse {
                error: "failed to fetch total memory for host".to_string(),
            }))
        }
    };

    let total_mem_bytes = match total_mem_bytes_str.parse::<i64>() {
        Ok(i) => i,
        Err(_) => {
            return Err(Json(ErrorResponse {
                error: "failed to parse total memory for host".to_string(),
            }))
        }
    };

    let usage_mem_bytes_str = match get_host_usage_mem(&hostid).await {
        Ok(mem) => mem,
        Err(_) => {
            return Err(Json(ErrorResponse {
                error: "failed to fetch used memory for host".to_string(),
            }))
        }
    };

    let usage_mem_bytes = match usage_mem_bytes_str.parse::<i64>() {
        Ok(i) => i,
        Err(_) => {
            return Err(Json(ErrorResponse {
                error: "failed to parse usage memory for host".to_string(),
            }))
        }
    };

    let cpu_total_str = match get_cpu_total(&hostid).await {
        Ok(mem) => mem,
        Err(_) => {
            return Err(Json(ErrorResponse {
                error: "failed to fetch total cpus for host".to_string(),
            }))
        }
    };

    let cpu_total = match cpu_total_str.parse::<i64>() {
        Ok(i) => i,
        Err(_) => {
            return Err(Json(ErrorResponse {
                error: "failed to parse total cpu for host".to_string(),
            }))
        }
    };

    let cpu_usage_str = match get_cpu_usage(&hostid).await {
        Ok(mem) => mem,
        Err(_) => {
            return Err(Json(ErrorResponse {
                error: "failed to fetch cpu usage for host".to_string(),
            }))
        }
    };

    let cpu_usage = match cpu_usage_str.parse::<i64>() {
        Ok(i) => i,
        Err(_) => {
            return Err(Json(ErrorResponse {
                error: "failed to parse cpu usage for host".to_string(),
            }))
        }
    };

    let powerstate_str = match get_state(&hostid).await {
        Ok(mem) => mem,
        Err(_) => {
            return Err(Json(ErrorResponse {
                error: "failed to fetch powerstate for host".to_string(),
            }))
        }
    };

    let powerstate = match powerstate_str.parse::<i64>() {
        Ok(i) => i,
        Err(_) => {
            return Err(Json(ErrorResponse {
                error: "failed to parse powerstate for host".to_string(),
            }))
        }
    };

    let vms = match get_state(&hostid).await {
        Ok(mem) => mem,
        Err(_) => {
            return Err(Json(ErrorResponse {
                error: "failed to fetch used cpus for host".to_string(),
            }))
        }
    };

    let mut host_states = match GLOBAL_HOST_STATE.write() {
        Ok(states) => states,
        Err(_) => {
            return Err(Json(ErrorResponse {
                error: "internal server error".to_string(),
            }))
        }
    };

    let host_state = match host_states.entry(hostid.clone()) {
        std::collections::hash_map::Entry::Occupied(o) => o.get().clone(),
        std::collections::hash_map::Entry::Vacant(v) => {
            let default_state = HostState {
                renewable_energy: false,
            };
            v.insert(default_state).clone()
        }
    };

    Ok(Json(HostInfo {
        state: host_state,
        total_mem_bytes,
        usage_mem_bytes,
        cpu_total,
        cpu_usage,
        powerstate,
        vms,
    }))
}
