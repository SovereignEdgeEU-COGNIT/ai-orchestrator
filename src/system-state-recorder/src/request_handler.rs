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

#[post("/hosts/<hostid>")]
pub async fn add_host(
    hosts: &State<Arc<Mutex<HashMap<String, SimHost>>>>,
    hostid: Option<String>,
) -> Result<Json<Response>, (Status, Json<ErrorResponse>)> {
    if let Some(host_id) = hostid {
        let mut host_guarded = hosts
            .inner()
            .lock()
            .expect("failed to lock the shared simulator");

        if host_guarded.contains_key(&host_id) {
            return Err((
                Status::BadRequest,
                Json(ErrorResponse {
                    error: "a host with the specified id already exists".to_string(),
                }),
            ));
        }

        let host = SimHost {
            hostid: host_id,
            vms: Vec::new(),
        };
        host_guarded.insert(host.hostid.clone(), host);

        return Ok(Json(Response {
            status: "success".to_string(),
            message: "host added successfully".to_string(),
        }));
    } else {
        return Err((
            Status::BadRequest,
            Json(ErrorResponse {
                error: "hostid is required".to_string(),
            }),
        ));
    }
}

#[post("/hosts/<hostid>/vms/<vmsid>")]
pub async fn add_vm(
    hosts: &State<Arc<Mutex<HashMap<String, SimHost>>>>,
    hostid: Option<String>,
    vmsid: Option<String>,
) -> Result<Json<Response>, (Status, Json<ErrorResponse>)> {
    if let Some(host_id) = hostid {
        if let Some(vm_id) = vmsid {
            let mut host_guarded = hosts
                .inner()
                .lock()
                .expect("failed to lock the shared simulator");

            if let Some(host) = host_guarded.get_mut(&host_id) {
                if host.vms.iter().any(|vm| vm.vmid == vm_id) {
                    return Err((
                        Status::BadRequest,
                        Json(ErrorResponse {
                            error: "a vm with the specified id already exists".to_string(),
                        }),
                    ));
                }

                let vm = VM { vmid: vm_id };
                host.vms.push(vm);

                return Ok(Json(Response {
                    status: "success".to_string(),
                    message: "vm added successfully".to_string(),
                }));
            } else {
                return Err((
                    Status::BadRequest,
                    Json(ErrorResponse {
                        error: "specified hostid does not exist".to_string(),
                    }),
                ));
            }
        } else {
            return Err((
                Status::BadRequest,
                Json(ErrorResponse {
                    error: "vmsid is required".to_string(),
                }),
            ));
        }
    } else {
        return Err((
            Status::BadRequest,
            Json(ErrorResponse {
                error: "hostid is required".to_string(),
            }),
        ));
    }
}

#[derive(FromForm)]
pub struct VmDeployQuery {
    vmid: Option<String>,
    mem: Option<String>,
    cpu: Option<String>,
    disk: Option<String>,
}

#[post("/deploy?<query_params..>")]
pub async fn place_vm(
    hosts: &State<Arc<Mutex<HashMap<String, SimHost>>>>,
    aiorchestrator_url: &State<Arc<String>>,
    query_params: VmDeployQuery,
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
    if query_params.disk.is_none() {
        return Err((
            Status::BadRequest,
            Json(ErrorResponse {
                error: "disk is not provided".to_string(),
            }),
        ));
    }

    let json_data = {
        let mut hosts_guarded = hosts
            .inner()
            .lock()
            .expect("failed to lock the shared simulator");
        let vmid: i32 = query_params
            .vmid
            .as_ref()
            .and_then(|s| s.parse().ok())
            .unwrap_or_default();
        let cpu: f32 = query_params
            .cpu
            .as_ref()
            .and_then(|s| s.parse().ok())
            .unwrap_or_default();
        let memory: i32 = query_params
            .mem
            .as_ref()
            .and_then(|s| s.parse().ok())
            .unwrap_or_default();
        let disk_size: i32 = query_params
            .disk
            .as_ref()
            .and_then(|s| s.parse().ok())
            .unwrap_or_default();
        let host_ids_vec: Vec<i32> = hosts_guarded
            .values()
            .filter_map(|host| host.hostid.parse::<i32>().ok()) // use filter_map to only keep successful parses
            .collect();

        let host_ids: Vec<String> = hosts_guarded.keys().cloned().collect(); // Collect the keys
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

        generate_placement_request_json(cpu, disk_size, memory, vmid, host_ids_vec)
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

            if let Some(host) = hosts_guarded.get_mut(&host_id_str) {
                let vm = VM {
                    vmid: vm.ID.to_string(),
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
