#!/usr/bin/env python3
import time

class MyTimer(object):
    """MyTimer simple timer that tracks lap and total ellapsed time"""

    def __init__(self):
        super(MyTimer, self).__init__()

    def start(self):
        self.start_time = time.time()
        self.current_time = self.start_time
        self.lap_time = self.start_time

    def lap(self):
        self.current_time = time.time()
        delta = self.current_time - self.lap_time
        self.lap_time = self.current_time
        return delta

    def ellapsed(self):
        return (time.time() - self.start_time)
