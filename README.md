# LongRaTGen
LongRaTGen stands for Long Random Trajectory Generator. Its goal is to generate long trajectories for Robot-Arms.
Wanted Features:
- Generate certain Types of Traj. (sinus, harsh start/stop, Random Parameters)
- output to txt file in a certain format thats easy to use
- output as VAL3 Code to use with Stäubli bots
- math based custom input functions woudl be sick



 ## Example
```python
from RaTGen import *
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

rt = RaTGen()
rt.set_dt(.01)

fun = lambda x: 0.1*x

x = rt.generate_sin(1, 1)
y = rt.generate_cos(1, 1)
z = rt.generate_custom(fun, 0, 2*np.pi) + rt.generate_noise(.01, 0, 2*np.pi)

ax.plot3D(x, y, z)

plt.show()
```python
![image](https://github.com/user-attachments/assets/0a7dab7f-a626-43a5-bec9-c97150f0ebd6)



![image](https://github.com/user-attachments/assets/68eb35e9-ec0e-49f5-b991-0ba8d528451a)
