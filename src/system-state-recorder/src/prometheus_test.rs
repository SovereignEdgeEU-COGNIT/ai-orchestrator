#[cfg(test)]
mod tests {
    use crate::prometheus::{generate_host_vm_map, get_host, get_hosts, get_vms, get_vms_for_host};

    #[tokio::test]
    async fn test_mapping_vms_to_hosts() {
        println!("printing all known hosts:");
        let host_ids = get_hosts().await.expect("failed to get hosts");
        assert!(!host_ids.is_empty(), "host IDs should not be empty");
        for host_id in host_ids.keys() {
            println!(" host_id={} value={}", host_id, host_ids[host_id]);
        }

        println!("printing all vms:");
        let vm_ids = get_vms().await.expect("failed to get VMs");
        assert!(!vm_ids.is_empty(), "VM IDs should not be empty");
        for vm_id in vm_ids.keys() {
            println!(" vm_id={}, value={}", vm_id, vm_ids[vm_id]);
        }

        println!("mapping vm to hosts:");
        for vm_id in vm_ids.keys() {
            let host_id = get_host(vm_id).await.expect("failed to get host for VM");
            match host_id {
                Some(host_id) => println!("vm_id={} is running on host_id={}", vm_id, host_id),
                None => println!("failed to retrieve host ID"),
            }
        }

        println!("mapping hosts to vms:");
        let host_to_vms_map = generate_host_vm_map(&vm_ids)
            .await
            .expect("failed to generate vm map");
        for host_id in host_ids.keys() {
            match get_vms_for_host(host_id, &host_ids, &host_to_vms_map).await {
                Ok(vms) => {
                    for vm_id in vms {
                        println!("host_id={} is running vm_id={}", host_id, vm_id);
                    }
                }
                Err(e) => println!("failed to get VM for host_id={}, error: {}", host_id, e),
            }
        }
    }
}
