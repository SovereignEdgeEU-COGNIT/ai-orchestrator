import random

class Server:
    def __init__(self, name, clock, cores=1, mem=10.0):
        self._name = name
        self._mem = mem
        self._clock = clock
        self._cores = cores
        self._cpuload = 0.0
        self._memload = 0.0
        self._processes = {} 

    @property
    def name(self):
        return self._name
    
    @property
    def cpuload(self):
        return self._cpuload

    def launch(self, process):
        pid = random.randint(1, 10**30 - 1)
        process.set_pid(pid)
        self._processes[pid] = process

    def execute(self):
        now = self._clock.get_time()
        self._cpuload = 0
        completed_processes = []
        for pid in self._processes.keys():
            process = self._processes[pid]
            if process.started is False:
                process.start(now)

            if process.duration(now) > process.exectime:
                print("process <{}> completed".format(process.pid))
                completed_processes.append(process.pid)
                continue

            process.execute()

            self._cpuload += process.cpuload / self._cores * 0.3
            if self._cpuload > 1.0:
                self._cpuload = 1
        
        for pid in completed_processes:
            del self._processes[pid]
