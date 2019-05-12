import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy import signal
ENV = "acrobot"
NAME = "results/" + ENV + ".png"

EXPERT = -96.85
CLONE = -101.93
RANDOM = -499.77

def getdata(name):
    all_results = []

    for i in range(0, 50):
        f = open(name + str(i) + ".csv", "r")
        results = []
        time = []

        for line in f:
            step, reward = (line.replace("\n", "")).split(",")
            results.append(float(reward))
            time.append(float(step))
        results = signal.savgol_filter(results, 7, 3, axis=-1)
        all_results.append(results)

    return all_results, time


# Load the data.
sns.set(rc={'figure.figsize':(10,9)})

results_1, time_1 = getdata("results/" + ENV + "_bco/")
results_2, time_2 = getdata("results/" + ENV + "/")

fig = plt.figure()
# Plot each line.
sns.tsplot(
    condition='ILPO', time=time_2, data=results_2, color='m', linestyle='-', err_style='ci_band', interpolate=True, legend=True)

sns.tsplot(
    condition='BCO', time=time_1, data=results_1, color='b', linestyle='-', err_style='ci_band', interpolate=True, legend=True)

plt.ylabel("Average Episodic Reward",fontsize=30)
plt.xlabel("Training Steps",fontsize=30)
plt.axhline(y=EXPERT, color='k', linestyle='--', label="Expert")
plt.axhline(y=CLONE, color='r', linestyle='--', label="BC")
plt.axhline(y=RANDOM, color='c', linestyle='--', label="Random")
plt.ylim([-510, -50])
plt.legend(prop={'size': 20}, loc="center right")

plt.savefig(NAME)
plt.show()
