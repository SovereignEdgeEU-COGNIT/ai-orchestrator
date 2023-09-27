#[cfg(test)]
mod tests {
    use crate::monitor::Monitor;
    use crate::prometheus::PrometheusMonitor;

    #[tokio::test]
    async fn test_mapping_vms_to_hosts() {
        let monitor = PrometheusMonitor;
        println!("printing all known hosts:");
        let host_ids = monitor.get_hosts().await.expect("failed to get hosts");
        assert!(!host_ids.is_empty(), "host IDs should not be empty");
        for host_id in host_ids.keys() {
            println!(" host_id={} value={}", host_id, host_ids[host_id]);
        }

        println!("printing all vms:");
        let vm_ids = monitor.get_vms().await.expect("failed to get VMs");
        assert!(!vm_ids.is_empty(), "VM IDs should not be empty");
        for vm_id in vm_ids.keys() {
            println!(" vm_id={}, value={}", vm_id, vm_ids[vm_id]);
        }

        println!("mapping vm to hosts:");
        for vm_id in vm_ids.keys() {
            let host_id = monitor
                .get_host(vm_id)
                .await
                .expect("failed to get host for VM");
            match host_id {
                Some(host_id) => println!("vm_id={} is running on host_id={}", vm_id, host_id),
                None => println!("failed to retrieve host ID"),
            }
        }

        println!("mapping hosts to vms:");
        let host_to_vms_map = monitor
            .generate_host_vm_map(&vm_ids)
            .await
            .expect("failed to generate vm map");
        for host_id in host_ids.keys() {
            match monitor
                .get_vms_for_host(host_id, &host_ids, &host_to_vms_map)
                .await
            {
                Ok(vms) => {
                    for vm_id in vms {
                        println!("host_id={} is running vm_id={}", host_id, vm_id);
                    }
                }
                Err(e) => println!("failed to get VM for host_id={}, error: {}", host_id, e),
            }
        }
        for host_id in host_ids.keys() {
            let mem = monitor
                .get_host_total_mem(host_id)
                .await
                .expect("failed to get host memory");
            println!("host_id={} has total mem={} bytes", host_id, mem);
        }
    }
}
