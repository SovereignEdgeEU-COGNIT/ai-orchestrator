import csv
import json
from io import StringIO

# Simulating loading your data
csv_data = """id,type,ts,cpu,memory,disk_read,disk_write,net_rx,net_tx
Cognit-test_emu_sr_memory_1026,0,2024-03-12 19:19:56.511000+00:00,0.21553673600277296,1181057024,0.0,0.0,0.09389298374443203,0.08423064040507512
Cognit-test_emu_sr_memory_1027,0,2024-03-12 19:19:56.716000+00:00,0.12843187740583914,780226560,0.0,0.0,0.07746792799210309,0.06943662096908847
Cognit-test_emu_sr_filesystem_1028,0,2024-03-12 19:19:56.909000+00:00,0.07938298988385162,17633280,0.0,0.0,0.13475757540277258,0.12195139799550393
Cognit-test-2_emu_sr_cpu_1024,0,2024-03-12 19:19:57.095000+00:00,0.2817444986449864,4308992,0.0,0.0,0.0964741418247516,0.08994608175248418
Cognit-test_emu_sr_network_1029,0,2024-03-12 19:19:57.334000+00:00,0.014636778172304243,92389376,0.0,0.0,0.13629169781349282,0.12282312885441975
Cognit-test-2_emu_sr_memory_1026,0,2024-03-12 19:19:57.643000+00:00,0.20448650087506526,1282904064,0.0,0.0,0.11192418702288552,0.09883348400033852
Cognit-test-2_emu_sr_memory_1027,0,2024-03-12 19:19:57.867000+00:00,0.12524473973135108,3964928,0.0,0.0,0.0942902041153169,0.08301996222124414
Cognit-test-2_emu_sr_filesystem_1028,0,2024-03-12 19:19:58.055000+00:00,0.03916741663782162,15527936,0.0,0.0,0.1425350827852775,0.12676044911734163
Cognit-test_emu_sr_memory_1026_,1,2024-03-12 19:19:58.508000+00:00,0.22172860819215093,1342177280,0.0,0.0,0.0,0.0
Cognit-test_emu_sr_memory_1027_,1,2024-03-12 19:19:58.703000+00:00,0.14407615991738587,507637760,0.0,0.0,0.0,0.0
Cognit-test_emu_sr_filesystem_1028_,1,2024-03-12 19:19:59.086000+00:00,0.07175172052663076,17633280,0.0,0.0,0.0,0.0
Cognit-test-2_emu_sr_cpu_1024_,1,2024-03-12 19:19:59.319000+00:00,0.26109081761006286,4304896,0.0,0.0,0.0,0.0
Cognit-test_emu_sr_network_1029_,1,2024-03-12 19:19:59.556000+00:00,0.014694124078159404,101171200,0.0,0.0,0.0,0.0
Cognit-test-2_emu_sr_memory_1026_,1,2024-03-12 19:19:59.887000+00:00,0.2010921013936325,1566437376,0.0,0.0,0.0,0.0
Cognit-test-2_emu_sr_memory_1027_,1,2024-03-12 19:20:00.130000+00:00,0.12444516845642763,3964928,0.0,0.0,0.0,0.0
Cognit-test-2_emu_sr_filesystem_1028_,1,2024-03-12 19:20:00.384000+00:00,0.03796774552742929,15527936,0.0,0.0,0.0,0.0
Cognit-test_emu_sr_memory_1026,0,2024-03-12 19:20:02.223000+00:00,0.21920686791362745,561623040,0.0,0.0,0.10938459142774608,0.09612585307286776
Cognit-test_emu_sr_memory_1027,0,2024-03-12 19:20:02.520000+00:00,0.13012164343360233,4124672,0.0,0.0,0.109119933052091,0.09650988169479092
Cognit-test_emu_sr_filesystem_1028,0,2024-03-12 19:20:02.726000+00:00,0.0719835471479307,139001856,0.0,0.0,0.1436130781091451,0.12959689587561307
Cognit-test-2_emu_sr_cpu_1024,0,2024-03-12 19:20:02.917000+00:00,0.23831277779722118,227880960,0.0,0.0,0.12826931237531936,0.11678556898120604
Cognit-test_emu_sr_network_1029,0,2024-03-12 19:20:03.112000+00:00,0.013601820843712089,5120000,0.0,0.0,0.11466617330391833,0.10328589998339527
Cognit-test-2_emu_sr_memory_1026,0,2024-03-12 19:20:03.393000+00:00,0.2110848658318426,798019584,0.0,0.0,0.11243571109123435,0.10160442754919499
Cognit-test-2_emu_sr_memory_1027,0,2024-03-12 19:20:03.576000+00:00,0.1990634163103237,1342025728,0.0,0.0,0.10137726929978619,0.09101210531841783"""  # Your CSV data goes here
txt_data = """{"host_info": {"ip": "194.28.122.122", "name": "Cognit-test", "port": 8001}, "sr_env": {"mem": 1536, "cpu": 1.0}, "client_info": {"execution_time": 3, "flavor": "cpu", "request_rate": 5}}
{"host_info": {"ip": "194.28.122.122", "name": "Cognit-test", "port": 8001}, "sr_env": {"mem": 1280, "cpu": 0.9}, "client_info": {"execution_time": 7, "flavor": "cpu", "request_rate": 14}}
{"host_info": {"ip": "194.28.122.122", "name": "Cognit-test", "port": 8001}, "sr_env": {"mem": 1280, "cpu": 0.7}, "client_info": {"execution_time": 7, "flavor": "memory", "request_rate": 12}}
{"host_info": {"ip": "194.28.122.122", "name": "Cognit-test", "port": 8001}, "sr_env": {"mem": 768, "cpu": 0.5}, "client_info": {"execution_time": 7, "flavor": "memory", "request_rate": 21}}
{"host_info": {"ip": "194.28.122.122", "name": "Cognit-test", "port": 8001}, "sr_env": {"mem": 1024, "cpu": 1.0}, "client_info": {"execution_time": 5, "flavor": "filesystem", "request_rate": 10}}
{"host_info": {"ip": "194.28.122.123", "name": "Cognit-test2", "port": 8001}, "sr_env": {"mem": 1536, "cpu": 0.5}, "client_info": {"execution_time": 7, "flavor": "cpu", "request_rate": 16}}
{"host_info": {"ip": "194.28.122.122", "name": "Cognit-test", "port": 8001}, "sr_env": {"mem": 512, "cpu": 0.2}, "client_info": {"execution_time": 5, "flavor": "network", "request_rate": 10}}
{"host_info": {"ip": "194.28.122.123", "name": "Cognit-test2", "port": 8001}, "sr_env": {"mem": 512, "cpu": 0.4}, "client_info": {"execution_time": 5, "flavor": "cpu", "request_rate": 9}}
{"host_info": {"ip": "194.28.122.123", "name": "Cognit-test2", "port": 8001}, "sr_env": {"mem": 1536, "cpu": 0.6}, "client_info": {"execution_time": 7, "flavor": "memory", "request_rate": 12}}
{"host_info": {"ip": "194.28.122.123", "name": "Cognit-test2", "port": 8001}, "sr_env": {"mem": 1280, "cpu": 0.9}, "client_info": {"execution_time": 7, "flavor": "memory", "request_rate": 20}}
{"host_info": {"ip": "194.28.122.123", "name": "Cognit-test2", "port": 8001}, "sr_env": {"mem": 1280, "cpu": 0.8}, "client_info": {"execution_time": 5, "flavor": "filesystem", "request_rate": 15}}
{"host_info": {"ip": "194.28.122.123", "name": "Cognit-test2", "port": 8001}, "sr_env": {"mem": 512, "cpu": 1.0}, "client_info": {"execution_time": 9, "flavor": "network", "request_rate": 23}}"""  # Your TXT data goes here

# Step 1: Filter and transform CSV
filtered_csv = StringIO()
reader = csv.DictReader(StringIO(csv_data))
writer = csv.DictWriter(filtered_csv, fieldnames=['CPU usage [%]', 'Memory usage [KB]', 'Disk read throughput [KB/s]', 'Disk write throughput [KB/s]', 'Network received throughput [KB/s]', 'Network transmitted throughput [KB/s]'])

writer.writeheader()
for row in reader:
    if not row['id'].endswith('_'):
        row['Memory usage [KB]'] = int(row['memory']) // 1024
        writer.writerow({new_key: row[old_key] for old_key, new_key in zip(reader.fieldnames[3:], writer.fieldnames)})

# Step 2: Parse TXT data
txt_lines = txt_data.splitlines()
host_info_mapping = {}
for line in txt_lines:
    data = json.loads(line)
    name =  "Cognit-test" if data['host_info']['name'] == "Cognit-test" else  "Cognit-test-2"
    name = name + "_emu_sr_" + data['client_info']['flavor'] + "_"
    key = name
    host_info_mapping[key] = {'cores': data['sr_env']['cpu'], 'total mem': data['sr_env']['mem']}

print(host_info_mapping)


# Assuming we can now map and integrate the TXT data with the filtered CSV, as needed
