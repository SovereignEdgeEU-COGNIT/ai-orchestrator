extern crate rocket;
extern crate serde;

use crate::monitor::Monitor;
use crate::placement_request::*;
use crate::placement_response::*;
use crate::simulator::Host as SimHost;
use crate::simulator::VM;
use rocket::http::Status;
use rocket::serde::json::Json;
use rocket::State;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
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
pub struct Response {
    status: String,
    message: String,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    error: String,
}

#[derive(Serialize)]
pub struct HostInfo {
    state: HostState,
    total_mem_bytes: i64,
    usage_mem_bytes: i64,
    cpu_total: i64,
    cpu_usage: i64,
    powerstate: i64,
    vms: i64,
}

#[derive(FromForm)]
pub struct VMDeployQuery {
    vmid: Option<String>,
    mem: Option<String>,
    cpu: Option<String>,
}

#[derive(FromForm)]
pub struct HostAddQuery {
    hostid: Option<String>,
    mem: Option<String>,
    cpu: Option<String>,
}

#[get("/")]
pub async fn index(
    monitor: &State<Arc<dyn Monitor + Send>>,
) -> Result<Json<Vec<Host>>, rocket::http::Status> {
    let host_ids = match monitor.get_hosts().await {
        Ok(hosts) => hosts,
        Err(_) => return Err(rocket::http::Status::InternalServerError),
    };

    let vm_ids = match monitor.get_vms().await {
        Ok(hosts) => hosts,
        Err(_) => return Err(rocket::http::Status::InternalServerError),
    };

    let mut mappings: Vec<Host> = Vec::new();

    let host_to_vms_map = match monitor.generate_host_vm_map(&vm_ids).await {
        Ok(m) => m,
        Err(_) => return Err(rocket::http::Status::InternalServerError),
    };

    for host_id in host_ids.keys() {
        let vm_ids = match monitor
            .get_vms_for_host(host_id, &host_ids, &host_to_vms_map)
            .await
        {
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

#[get("/set?<hostid>&<renewable>")]
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
pub async fn get_host_info(
    monitor: &State<Arc<dyn Monitor + Send>>,
    hostid: String,
) -> Result<Json<HostInfo>, Json<ErrorResponse>> {
    if !monitor
        .check_host_exists(&hostid)
        .await
        .map_err(|_| ErrorResponse {
            error: "internal server error".to_string(),
        })?
    {
        return Err(Json(ErrorResponse {
            error: "host not found".to_string(),
        }));
    }

    let total_mem_bytes_str = match monitor.get_host_total_mem(&hostid).await {
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

    let usage_mem_bytes_str = match monitor.get_host_usage_mem(&hostid).await {
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

    let cpu_total_str = match monitor.get_cpu_total(&hostid).await {
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

    let cpu_usage_str = match monitor.get_cpu_usage(&hostid).await {
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

    let powerstate_str = match monitor.get_state(&hostid).await {
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

    let vms_str = match monitor.get_host_vms(&hostid).await {
        Ok(mem) => mem,
        Err(_) => {
            return Err(Json(ErrorResponse {
                error: "failed to fetch used cpus for host".to_string(),
            }))
        }
    };

    let vms = match vms_str.parse::<i64>() {
        Ok(i) => i,
        Err(_) => {
            return Err(Json(ErrorResponse {
                error: "failed to parse total cpu for host".to_string(),
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

#[post("/addhost?<query_params..>")]
pub async fn add_host(
    hosts: &State<Arc<Mutex<HashMap<String, SimHost>>>>,
    query_params: HostAddQuery,
) -> Result<Json<Response>, (Status, Json<ErrorResponse>)> {
    if query_params.hostid.is_none() {
        return Err((
            Status::BadRequest,
            Json(ErrorResponse {
                error: "hostid is not provided".to_string(),
            }),
        ));
    }
    if query_params.mem.is_none() {
        return Err((
            Status::BadRequest,
            Json(ErrorResponse {
                error: "mem is not provided".to_string(),
            }),
        ));
    }
    if query_params.cpu.is_none() {
        return Err((
            Status::BadRequest,
            Json(ErrorResponse {
                error: "cpu is not provided".to_string(),
            }),
        ));
    }
    let hostid: i64 = query_params
        .hostid
        .as_ref()
        .and_then(|s| s.parse().ok())
        .unwrap_or_default();
    let cpu: i64 = query_params
        .cpu
        .as_ref()
        .and_then(|s| s.parse().ok())
        .unwrap_or_default();
    let mem: i64 = query_params
        .mem
        .as_ref()
        .and_then(|s| s.parse().ok())
        .unwrap_or_default();
    {
        let mut host_guarded = hosts
            .inner()
            .lock()
            .expect("failed to lock the shared simulator");

        if host_guarded.contains_key(&hostid.to_string()) {
            return Err((
                Status::BadRequest,
                Json(ErrorResponse {
                    error: "a host with the specified id already exists".to_string(),
                }),
            ));
        }

        let host = SimHost {
            hostid: hostid.to_string(),
            vms: Vec::new(),
            mem: mem,
            cpu: cpu,
        };
        host_guarded.insert(host.hostid.clone(), host);

        return Ok(Json(Response {
            status: "success".to_string(),
            message: "host added successfully".to_string(),
        }));
    }
}

fn check_cpu_usage(
    hosts_guarded: &MutexGuard<'_, HashMap<String, SimHost>>,
    host_id_str: &String,
    cpu: i64,
) -> Result<(), (Status, Json<ErrorResponse>)> {
    let cpu_total = match hosts_guarded.get(host_id_str) {
        Some(host) => {
            let total_cpu_used = host.cpu;
            Ok(total_cpu_used.to_string())
        }
        None => Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "hostid not found",
        ))),
    };

    let t: String = match &cpu_total {
        Ok(usage) => usage.clone(),
        Err(_) => "error".to_string(),
    };

    let cpu_usage = match hosts_guarded.get(host_id_str) {
        Some(host) => {
            let total_cpu_used: i64 = host.vms.iter().map(|vm| vm.cpu).sum();
            Ok(total_cpu_used.to_string())
        }
        None => Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "hostid not found",
        ))),
    };

    let u: String = match &cpu_usage {
        Ok(usage) => usage.clone(),
        Err(_) => "error".to_string(),
    };

    let t_num = match t.parse::<i64>() {
        Ok(value) => value,
        Err(_) => {
            return Err((
                Status::BadRequest,
                Json(ErrorResponse {
                    error: "Failed to parse t".to_string(),
                }),
            ))
        }
    };

    let u_num = match u.parse::<i64>() {
        Ok(value) => value,
        Err(_) => {
            return Err((
                Status::BadRequest,
                Json(ErrorResponse {
                    error: "Failed to parse u".to_string(),
                }),
            ))
        }
    };

    if u_num + cpu > t_num {
        return Err((
            Status::BadRequest,
            Json(ErrorResponse {
                error: "too few cpus available".to_string(),
            }),
        ));
    }

    Ok(())
}

fn check_mem_usage(
    hosts_guarded: &MutexGuard<'_, HashMap<String, SimHost>>,
    host_id_str: &String,
    mem: i64,
) -> Result<(), (Status, Json<ErrorResponse>)> {
    let mem_total = match hosts_guarded.get(host_id_str) {
        Some(host) => {
            let total_mem_used = host.mem;
            Ok(total_mem_used.to_string())
        }
        None => Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "hostid not found",
        ))),
    };

    let t: String = match &mem_total {
        Ok(usage) => usage.clone(),
        Err(_) => "error".to_string(),
    };

    let mem_usage = match hosts_guarded.get(host_id_str) {
        Some(host) => {
            let total_mem_used: i64 = host.vms.iter().map(|vm| vm.mem).sum();
            Ok(total_mem_used.to_string())
        }
        None => Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "hostid not found",
        ))),
    };

    let u: String = match &mem_usage {
        Ok(usage) => usage.clone(),
        Err(_) => "error".to_string(),
    };

    let t_num = match t.parse::<i64>() {
        Ok(value) => value,
        Err(_) => {
            return Err((
                Status::BadRequest,
                Json(ErrorResponse {
                    error: "Failed to parse t".to_string(),
                }),
            ))
        }
    };

    let u_num = match u.parse::<i64>() {
        Ok(value) => value,
        Err(_) => {
            return Err((
                Status::BadRequest,
                Json(ErrorResponse {
                    error: "Failed to parse u".to_string(),
                }),
            ))
        }
    };

    if u_num + mem > t_num {
        return Err((
            Status::BadRequest,
            Json(ErrorResponse {
                error: "not enough memory available".to_string(),
            }),
        ));
    }

    Ok(())
}

#[post("/placevm?<query_params..>")]
pub async fn place_vm(
    hosts: &State<Arc<Mutex<HashMap<String, SimHost>>>>,
    aiorchestrator_url: &State<Arc<String>>,
    query_params: VMDeployQuery,
) -> Result<Json<Response>, (Status, Json<ErrorResponse>)> {
    print!("{}", &**aiorchestrator_url);
    if query_params.vmid.is_none() {
        return Err((
            Status::BadRequest,
            Json(ErrorResponse {
                error: "vmid is not provided".to_string(),
            }),
        ));
    }
    if query_params.mem.is_none() {
        return Err((
            Status::BadRequest,
            Json(ErrorResponse {
                error: "mem is not provided".to_string(),
            }),
        ));
    }
    if query_params.cpu.is_none() {
        return Err((
            Status::BadRequest,
            Json(ErrorResponse {
                error: "cpu is not provided".to_string(),
            }),
        ));
    }

    let cpu: i64 = query_params
        .cpu
        .as_ref()
        .and_then(|s| s.parse().ok())
        .unwrap_or_default();
    let mem: i64 = query_params
        .mem
        .as_ref()
        .and_then(|s| s.parse().ok())
        .unwrap_or_default();
    let vmid: i64 = query_params
        .vmid
        .as_ref()
        .and_then(|s| s.parse().ok())
        .unwrap_or_default();

    let json_data = {
        let mut hosts_guarded = hosts
            .inner()
            .lock()
            .expect("failed to lock the shared simulator");
        let host_ids_vec: Vec<i32> = hosts_guarded
            .values()
            .filter_map(|host| host.hostid.parse::<i32>().ok())
            .collect();

        if host_ids_vec.is_empty() {
            return Err((
                Status::BadRequest,
                Json(ErrorResponse {
                    error: "cannot send placement request, no host is available".to_string(),
                }),
            ));
        }

        let host_ids: Vec<String> = hosts_guarded.keys().cloned().collect();
        for host_id in host_ids {
            if let Some(host) = hosts_guarded.get_mut(&host_id) {
                if host.vms.iter().any(|vm| vm.vmid == vmid.to_string()) {
                    return Err((
                        Status::BadRequest,
                        Json(ErrorResponse {
                            error: "a vm with the specified id already exists".to_string(),
                        }),
                    ));
                }
            }
        }

        let disk_size = 0;
        generate_placement_request_json(cpu, disk_size, mem, vmid, host_ids_vec)
    };

    let client = reqwest::Client::new();
    let response = client
        .post((**aiorchestrator_url).as_str())
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(json_data.clone())
        .send()
        .await
        .map_err(|_| {
            (
                Status::BadRequest,
                Json(ErrorResponse {
                    error: "failed to connect to the ai orchestrator".to_string(),
                }),
            )
        })?;

    let text = response.text().await.map_err(|_| {
        (
            Status::BadRequest,
            Json(ErrorResponse {
                error: "error reading response".to_string(),
            }),
        )
    })?;

    let data = parse_placement_response_json(text.as_str()).map_err(|e| {
        println!("error parsing JSON: {:?}", e);
        (
            Status::BadRequest,
            Json(ErrorResponse {
                error: "error parsing the JSON response".to_string(),
            }),
        )
    })?;

    {
        let mut hosts_guarded = hosts
            .inner()
            .lock()
            .expect("failed to lock the shared simulator");
        for vm in &data.VMS {
            let host_id_str = vm.HOST_ID.to_string();

            check_cpu_usage(&hosts_guarded, &host_id_str.clone(), cpu)?;
            check_mem_usage(&hosts_guarded, &host_id_str.clone(), mem)?;

            if let Some(host) = hosts_guarded.get_mut(&host_id_str) {
                let vm = VM {
                    vmid: vm.ID.to_string(),
                    mem: mem,
                    cpu: cpu,
                };

                host.vms.push(vm);
            } else {
                return Err((
                    Status::BadRequest,
                    Json(ErrorResponse {
                        error: "specified hostid does not exist".to_string(),
                    }),
                ));
            }
        }
    }

    let msg = data
        .VMS
        .iter()
        .map(|vm| format!("vmid={} was deployed at hostid={}", vm.ID, vm.HOST_ID))
        .collect::<Vec<String>>()
        .join(", ");

    Ok(Json(Response {
        status: "success".to_string(),
        message: msg,
    }))
}

#[get("/issim")]
pub fn is_sim(is_sim: &State<Arc<bool>>) -> Result<Json<bool>, Status> {
    match **is_sim.inner() {
        true => Ok(Json(true)),
        false => Ok(Json(false)),
    }
}

#[delete("/vms/<vmid>")]
pub async fn delete_vm(
    hosts: &State<Arc<Mutex<HashMap<String, SimHost>>>>,
    vmid: String,
) -> Result<Json<Response>, (Status, Json<ErrorResponse>)> {
    let mut hosts_guarded = hosts
        .inner()
        .lock()
        .expect("failed to lock the shared simulator");

    for (_host_id, host) in hosts_guarded.iter_mut() {
        if let Some(vm_index) = host.vms.iter().position(|vm| vm.vmid == vmid) {
            host.vms.remove(vm_index);

            return Ok(Json(Response {
                status: "success".to_string(),
                message: format!("vm with id {} removed successfully", vmid),
            }));
        }
    }

    Err((
        Status::NotFound,
        Json(ErrorResponse {
            error: format!("vm with id {} not found", vmid),
        }),
    ))
}

#[derive(Serialize)]
pub struct VMResponse {
    hostid: String,
    vm: VM,
}

#[get("/vms/<vmid>")]
pub async fn get_vm(
    hosts: &State<Arc<Mutex<HashMap<String, SimHost>>>>,
    vmid: String,
) -> Result<Json<VMResponse>, (Status, Json<ErrorResponse>)> {
    let hosts_guarded = hosts
        .inner()
        .lock()
        .expect("Failed to lock the shared simulator");

    // Iterate over each host to find the VM with the specified vmid.
    for (host_id, host) in hosts_guarded.iter() {
        if let Some(vm) = host.vms.iter().find(|&v| v.vmid == vmid) {
            return Ok(Json(VMResponse {
                hostid: host_id.clone(),
                vm: vm.clone(),
            }));
        }
    }

    // If the VM with the specified vmid is not found in any host.
    Err((
        Status::NotFound,
        Json(ErrorResponse {
            error: format!("VM with id {} not found", vmid),
        }),
    ))
}
