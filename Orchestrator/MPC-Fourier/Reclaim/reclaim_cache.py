import time

recently_deleted_pods = {}  # pod_name -> timestamp
RECLAIM_GRACE_PERIOD = 60  # seconds

def is_recently_deleted(pod_name):
    now = time.time()
    ts = recently_deleted_pods.get(pod_name)
    return ts is not None and (now - ts) < RECLAIM_GRACE_PERIOD

def update_recent_deletions(pod_name):
    recently_deleted_pods[pod_name] = time.time()

def cleanup_deletion_history():
    now = time.time()
    keys_to_remove = [p for p, ts in recently_deleted_pods.items() if now - ts > RECLAIM_GRACE_PERIOD]
    for key in keys_to_remove:
        del recently_deleted_pods[key]