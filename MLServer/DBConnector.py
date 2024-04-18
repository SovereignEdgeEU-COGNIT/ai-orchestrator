

import os
from psycopg2 import pool
from typing import List, Optional
from dataclasses import dataclass

# Environment variables for DB connection
DB_HOST = os.getenv("ENVSERVER_DB_HOST")
DB_USER = os.getenv("ENVSERVER_DB_USER")
DB_PASSWORD = os.getenv("ENVSERVER_DB_PASSWORD")
DB_PORT = os.getenv("ENVSERVER_DB_PORT")

@dataclass
class Host:
    hostid: str
    stateid: int
    total_cpu: float
    total_mem: float
    usage_cpu: float
    usage_mem: float
    disk_read: float
    disk_write: float
    net_rx: float
    net_tx: float
    energy_usage: float

@dataclass
class Vm:
    vmid: str
    stateid: int
    deployed: bool
    hostid: str
    hoststateid: int
    total_cpu: float
    total_mem: float
    usage_cpu: float
    usage_mem: float
    disk_read: float
    disk_write: float
    net_rx: float
    net_tx: float

@dataclass
class Metric:
    id: str
    type: int
    ts: str  # Timestamp as string for simplicity
    cpu: float
    memory: float
    disk_read: float
    disk_write: float
    net_rx: float
    net_tx: float
    energy_usage: float

class DBClient:
    def __init__(self, minconn: int, maxconn: int):
        self.connection_pool = pool.ThreadedConnectionPool(
            minconn, 
            maxconn,
            dbname="postgres", 
            user=DB_USER, 
            password=DB_PASSWORD, 
            host=DB_HOST, 
            port=DB_PORT
        )
    
    def fetch_hosts(self) -> List[Host]:
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM PROD_HOSTS;")
                rows = cur.fetchall()
                return [Host(*row) for row in rows]
        finally:
            self.connection_pool.putconn(conn)
            
    def fetch_host(self, id: int) -> List[Host]:
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""SELECT * FROM PROD_HOSTS
                WHERE HOSTID = %s;
                """, (str(id),))
                rows = cur.fetchall()
                return [Host(*row) for row in rows]
        finally:
            self.connection_pool.putconn(conn)
            
    def fetch_host_vms(self, host_id: int) -> List[Vm]:
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""SELECT * FROM PROD_VMS
                WHERE HOSTID = %s;
                """, (str(host_id),))
                rows = cur.fetchall()
                return [Vm(*row) for row in rows]
        finally:
            self.connection_pool.putconn(conn)
    
    def fetch_vms(self) -> List[Vm]:
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM PROD_VMS;")
                rows = cur.fetchall()
                return [Vm(*row) for row in rows]
        finally:
            self.connection_pool.putconn(conn)
        
    def fetch_vm(self, id: int) -> List[Vm]:
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cur:
                #print(str(id))
                cur.execute("""
                SELECT * FROM PROD_VMS
                WHERE VMID = %s;
                """, (str(id),))
                rows = cur.fetchall()
                return [Vm(*row) for row in rows]
        finally:
            self.connection_pool.putconn(conn)
    
    def fetch_latest_metrics(self, id: int, metric_type: int, limit: int) -> List[Metric]:
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cur:
                query = """
                SELECT * FROM PROD_METRICS 
                WHERE ID = %s AND TYPE = %s 
                ORDER BY TS DESC 
                LIMIT %s;
                """
                cur.execute(query, (str(id), str(metric_type), str(limit)))
                rows = cur.fetchall()
                return [Metric(*row) for row in rows]
        finally:
            self.connection_pool.putconn(conn)
        
    def close_all_connections(self):
        self.connection_pool.closeall()


# Example usage
if __name__ == "__main__":
    db_client = DBClient(minconn=1, maxconn=10)
    try:
        hosts = db_client.fetch_hosts()
        vms = db_client.fetch_vms()
        latest_metrics = db_client.fetch_latest_metrics(1317, 1, 10)
        print(hosts)
        #print(vms)
        #print(latest_metrics)
    finally:
        db_client.close_all_connections()
    
    # Now, 'hosts', 'vms', and 'latest_metrics' contain the data fetched from the database.
