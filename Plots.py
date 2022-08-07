import matplotlib.pyplot as plt
import numpy as np

SBP_PATH = "./deep_learning/sbp/results.csv"
DBP_PATH = "./deep_learning/dbp/results.csv"

sbp = np.genfromtxt(SBP_PATH, delimiter = ',')[:30]
dbp = np.genfromtxt(DBP_PATH, delimiter = ',')[:30]

yLabel1 = []
yLabel2 = []

for epoch, loss in sbp:
    yLabel1.append(loss)

for epoch, loss in dbp:
    yLabel2.append(loss)


plt.plot(yLabel2, linestyle = 'dotted')
plt.show()