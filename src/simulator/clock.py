import time

class Clock:
    def __init__(self, nano_unix_time=None):
        # If nano_unix_time is None, set to the current Unix time in nanoseconds.
        self.nano_unix_time = nano_unix_time if nano_unix_time is not None else int(time.time() * 1e9)

    def tick_ns(self, nanoseconds=1):
        """Move time forward by the given number of nanoseconds."""
        self.nano_unix_time += nanoseconds

    def tick_micro(self, microseconds=1):
        """Move time forward by the given number of microseconds."""
        self.nano_unix_time += milliseconds * 1e3
    
    def tick_ms(self, milliseconds=1):
        """Move time forward by the given number of milliseconds."""
        self.nano_unix_time += milliseconds * 1e6
    
    def tick_s(self, seconds=1):
        """Move time forward by the given number of seconds."""
        self.nano_unix_time += seconds * 1e9

    def set_time(self, nano_unix_time):
        """Set to a specific Unix time in nanoseconds."""
        self.nano_unix_time = nano_unix_time

    def get_time(self):
        """Return the current Unix time in nanoseconds."""
        return self.nano_unix_time

    def display(self):
        """Display as a human-readable string."""
        seconds = self.nano_unix_time // 1e9  # Convert nanoseconds to seconds for display
        time_struct = time.gmtime(seconds)
        return time.strftime("%Y-%m-%d %H:%M:%S", time_struct)
