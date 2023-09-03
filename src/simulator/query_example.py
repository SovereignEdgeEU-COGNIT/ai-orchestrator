from prometheus_api_client import PrometheusConnect

prometheus_url = 'http://nauvoo:9090'
prometheus = PrometheusConnect(url=prometheus_url)

query = 'edgecluster_cpuload_rise'
result = prometheus.custom_query(query=query)
for item in result:
    print(item)
    metric_name = item['metric']['__name__']
    value = item['value'][1]
    print(f"Metric: {metric_name}, Value: {value}")
