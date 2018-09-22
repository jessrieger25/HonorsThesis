import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

plt.show()

import matplotlib.pyplot as plt
import numpy as np

def scatterplot(self, x_data, y_data, x_label="", y_label="", title="", color="r", yscale_log=False):
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y = [5, 2, 4, 2, 1, 4, 5, 2]

    plt.scatter(x, y, label='skitscat', color='k', s=25, marker="o")

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interesting Graph\nCheck it out')
    plt.legend()
    plt.show()

x = np.array([2,3,1,0])
y = np.array([2,3,1,0])
print(x)
print(y)
print(len(x), len(y))
scatterplot(x, y, 'x', 'y', 'test')
