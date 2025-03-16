import matplotlib.pyplot as plt
import numpy as np

categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5']
values = [4, 3, 6, 2, 7, 8]

fig = plt.figure()
ax = fig.add_subplot(111, polar=True)

angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

ax.plot(angles, values, color='b', linewidth=2, linestyle='solid', label="Data")
ax.fill(angles, values, color='b', alpha=0.25)

ax.set_thetagrids(angles[:-1] * 180/np.pi, categories)
plt.legend(loc='best')
plt.title("Radar Chart")

plt.show()