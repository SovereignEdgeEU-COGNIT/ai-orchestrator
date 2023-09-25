extern crate rocket;
extern crate serde;

use rocket::serde::json::Json;
use serde::Serialize;
use simulator::prometheus::{get_host, get_hosts, get_vms};

#[derive(Serialize)]
pub struct Host {
    hostid: String,
    vmids: Vec<String>,
}

#[get("/")]
pub async fn index() -> Result<Json<Vec<Host>>, rocket::http::Status> {
    let vms_ids = match get_vms().await {
        Ok(ids) => ids,
        Err(_) => return Err(rocket::http::Status::InternalServerError),
    };

    let mut mappings: Vec<Host> = Vec::new();

    for vm_id in vms_ids.keys() {
        let host_id = match get_host(vm_id).await {
            Ok(Some(id)) => id,
            _ => continue,
        };

        mappings.push(Host {
            hostid: host_id.clone(),
            vmids: vec![vm_id.clone()],
        });
    }

    Ok(Json(mappings))
}
