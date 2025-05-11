import numpy as np


class Load:
    x1, x2, x3, x4, y = np.loadtxt(
    "/home/ilya/Python/Deep/DL/dataset/data_reservasi.txt",
    skiprows= 1,
    unpack= True
    )
    
    @classmethod
    def get_data(cls):
        return cls.x1, cls.x2, cls.x3, cls.x4, cls.y

Load()