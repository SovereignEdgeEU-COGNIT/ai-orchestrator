#[cfg(test)]
mod tests {
    use crate::placement_request::*;

    #[tokio::test]
    async fn test_placement_request() {
        let host_ids = vec![0, 2, 3, 4];
        let cpu = 1.0;
        let disk_size = 2252;
        let memory = 786432;
        let vmid = 7;

        let json_output = generate_placement_request_json(cpu, disk_size, memory, vmid, host_ids);
        println!("{}", json_output);
    }
}
