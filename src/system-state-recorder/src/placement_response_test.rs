#[cfg(test)]
mod tests {
    use crate::placement_response::*;

    #[tokio::test]
    async fn test_placement_response() {
        let json_input = r#"{ "VMS": [{"ID": 7, "HOST_ID": 4}]}"#;

        match parse_placement_response_json(json_input) {
            Ok(data) => {
                for vm in &data.VMS {
                    println!("VM ID: {}, HOST ID: {}", vm.ID, vm.HOST_ID);
                }
            }
            Err(e) => {
                println!("Failed to parse JSON: {}", e);
            }
        }
    }
}
