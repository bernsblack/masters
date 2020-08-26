import numpy as np

from utils.configs import BaseConf
from utils.preprocessing import Shaper
conf = BaseConf()

data = np.random.randint(0,2,(1,1,100,100))
shaper = Shaper(data, conf)
print(shaper)

np.linalg.eig