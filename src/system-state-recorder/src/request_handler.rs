extern crate rocket;
extern crate serde;

use rocket::serde::json::Json;
use serde::Serialize;
use simulator::prometheus::{generate_host_vm_map, get_host, get_hosts, get_vms, get_vms_for_host};

#[derive(Serialize)]
pub struct Host {
    hostid: String,
    vmids: Vec<String>,
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

        mappings.push(Host {
            hostid: host_id.clone(),
            vmids: vm_ids,
        });
    }

    Ok(Json(mappings))
}
