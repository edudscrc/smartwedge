import numpy as np
from scipy.signal import gausspulse

class Transducer:
    def __init__(self, pitch: float=.4e-3, num_elem: int=64, element_gap:float=.1e-3, fc: float=5e6, bw: float=.4, bwr: float=-6,
                 pulse_type: str="gaussian"):
        self.pitch = pitch
        self.num_elem = num_elem
        self.fc = fc
        self.bw = bw
        self.bwr = bwr
        self.pulse_type = pulse_type
        self.xt = np.arange(0, self.num_elem) * pitch
        self.xt -= np.mean(self.xt)
        self.zt = np.zeros_like(self.xt)
        self.elements = np.arange(1, self.num_elem + 1, 1)
        self.element_gap = element_gap
        self.element_width = self.pitch - self.element_gap

    ######################
    ## TRANSDUCER UTILS ##
    ######################
    def get_coords(self, i: int = -1):
        if i == -1:
            return self.xt, self.zt
        else:
            return self.xt[i], self.zt[i]
