import numpy as np

#non linear least sqauares estimation and dynamics

class dynamics:
    def __init__(self, x, u, t_idx, time):
        self.x = x
        self.u = u
        self.t_idx = t_idx
        self.time = time
        self.v = []
        self.g = []
        self.h = []
        self.obs_x = []


class Data_idx:
    def __init__(self, u, t_idx, time):
        self.t_idx = t_idx
        self.time = time
        self.u_init = u
        self.g = []
        self.h = []
        self.x = []
        self.obs_x = []

def main():
