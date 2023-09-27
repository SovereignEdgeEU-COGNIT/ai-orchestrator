#[cfg(test)]
mod tests {
    use crate::{
        monitor::Monitor,
        simulator::{Simulator, SimulatorHelper},
    };

    #[tokio::test]
    async fn test_simulator() {
        let mut simulator = Simulator::new();
        simulator.add_host_with_vms("1".to_string(), vec!["1".to_string(), "2".to_string()]);
        simulator.add_host_with_vms("2".to_string(), vec!["3".to_string()]);
        let host_vms = simulator
            .get_host_vms(&"2".to_string())
            .await
            .expect("failed to get host VMs");
        println!("host_vms={}", host_vms);

        let hostid = simulator
            .get_host(&"3".to_string())
            .await
            .expect("failed to get host");
        println!("hostid={}", hostid.unwrap());

        println!("mapping hosts to vms:");
        let vm_ids = simulator.get_vms().await.expect("Failed to get VM ids");
        let host_ids = simulator.get_hosts().await.expect("Failed to get hosts id");

        let host_to_vms_map = simulator
            .generate_host_vm_map(&vm_ids)
            .await
            .expect("failed to generate vm map");
        for host_id in host_ids.keys() {
            match simulator
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
    }
}
