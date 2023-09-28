#![allow(non_snake_case)]

use serde::Serialize;
#[derive(Serialize)]
struct Capacity {
    CPU: i64,
    DISK_SIZE: i64,
    MEMORY: i64,
}

#[derive(Serialize)]
struct UserTemplate {
    LOGO: String,
    LXD_SECURITY_PRIVILEGED: String,
    SCHED_REQUIREMENTS: String,
}

#[derive(Serialize)]
struct VM {
    CAPACITY: Capacity,
    HOST_IDS: Vec<i32>,
    ID: i64,
    STATE: String,
    USER_TEMPLATE: UserTemplate,
}

#[derive(Serialize)]
struct Data {
    VMS: Vec<VM>,
}

pub fn generate_placement_request_json(
    cpu: i64,
    disk_size: i64,
    memory: i64,
    vmid: i64,
    host_ids: Vec<i32>,
) -> String {
    let requirements: Vec<String> = host_ids.iter().map(|id| format!("ID=\"{}\"", id)).collect();
    let sched_requirements = requirements.join(" | ");

    let data = Data {
        VMS: vec![VM {
            CAPACITY: Capacity {
                CPU: cpu,
                DISK_SIZE: disk_size,
                MEMORY: memory,
            },
            HOST_IDS: host_ids,
            ID: vmid,
            STATE: "PENDING".to_string(),
            USER_TEMPLATE: UserTemplate {
                LOGO: "images/logos/ubuntu.png".to_string(),
                LXD_SECURITY_PRIVILEGED: "true".to_string(),
                SCHED_REQUIREMENTS: sched_requirements,
            },
        }],
    };

    serde_json::to_string_pretty(&data).unwrap()
}
