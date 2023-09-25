#[cfg(test)]
mod tests {
    use crate::prometheus::{get_host, get_hosts, get_vms};

    #[tokio::test]
    async fn test_mapping_vms_to_hosts() {
        println!("printing all known hosts:");
        let host_ids = get_hosts().await.expect("Failed to get hosts");
        assert!(!host_ids.is_empty(), "Host IDs should not be empty");
        for host_id in host_ids.keys() {
            println!(" host_id={}", host_id);
        }

        println!("printing all vms:");
        let vms_ids = get_vms().await.expect("Failed to get VMs");
        assert!(!vms_ids.is_empty(), "VM IDs should not be empty");
        for vm_id in vms_ids.keys() {
            println!(" vm_id={}", vm_id);
        }

        println!("mapping vm to hosts:");
        for vm_id in vms_ids.keys() {
            let host_id = get_host(vm_id).await.expect("Failed to get host for VM");
            match host_id {
                Some(host_id) => println!("vm_id={} is running on host_id={}", vm_id, host_id),
                None => println!("Failed to retrieve host ID"),
            }
        }
    }
}
