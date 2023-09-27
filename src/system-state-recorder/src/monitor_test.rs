use crate::prometheus::PrometheusMonitor;

#[cfg(test)]
mod tests {
    use super::PrometheusMonitor;
    use crate::monitor::Monitor;

    #[tokio::test]
    async fn test_monitor() {
        let prom_monitor = PrometheusMonitor;

        let host_id = "2".to_string();
        match prom_monitor.get_cpu_usage(&host_id).await {
            Ok(cpu_usage) => {
                println!("CPU Usage for host {}: {}", host_id, cpu_usage);
            }
            Err(e) => {
                eprintln!("Error fetching CPU usage: {}", e);
            }
        }
    }
}
