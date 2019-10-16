import matplotlib.pyplot as plt
import numpy as np
import math


def gaussian(x, u, sigma):
    return np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)


# x = np.linspace(220, 230, 10000)
x = np.linspace(-11, 11, 23)
y = [gaussian(i, 0, 4) for i in np.linspace(-11, 11, 23)]
max_y = max(y)

# plt.title('PDF in Horizontal Direction', fontsize=22)
# plt.xticks(fontsize=22)
# plt.yticks(fontsize=22)
# axes = plt.subplot(111)
# axes.set_xticks([-800, -400, 0, 400, 800])
# axes.set_yticks([0, 0.001, 0.002, 0.0030])
plt.plot(x, y/max_y, "b-")
plt.show()
print('wow')