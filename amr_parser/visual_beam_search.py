# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-11-22 19:10
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


def to_scores(text):
    return [float(x) for x in text.split()]


giis = to_scores('80.2	80.2	80	80	77.5')
merge = to_scores('80	80.1	80	79.7	78.4')
levi = to_scores('80	80	79.9	79.4	78.3')

# evenly sampled time at 200ms intervals
t = [1, 2, 4, 8, 16]
t = list(reversed(t))

# red dashes, blue squares and green triangles
plt.rcParams["figure.figsize"] = (4, 4)
_, ax = plt.subplots()

# Be sure to only pick integer tick locations.
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot(t, giis, 'r--')
plt.plot(t, merge, 'g:')
plt.plot(t, levi, 'b')
plt.legend(['GSII', 'ND + AD + BD', 'ND + AD + Levi'])
plt.xlabel('Beam Size')
plt.ylabel('Smatch')
plt.savefig("/Users/hankcs/Dropbox/应用/Overleaf/NAACL-2021-AMR/fig/beam.pdf", bbox_inches='tight')
plt.show()
