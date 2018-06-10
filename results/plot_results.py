import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


FILE = "results/cartpole/"
NAME = "results/cartpole.png"

def getdata(name):
    all_results = []

    for i in range(0, 100):
        f = open(name + str(i) + ".csv", "r")
        results = []
        time = []

        for line in f:
            step, reward = (line.replace("\n", "")).split(",")
            results.append(float(reward))
            time.append(float(step))

        all_results.append(results)


    return all_results, time


# Load the data.
sns.set(rc={'figure.figsize':(10,9)})

results_1, time = getdata(FILE)
fig = plt.figure()

# Plot each line.
sns.tsplot(
    condition='ILPO', time=time, data=results_1, color='b', linestyle='-', err_style='ci_band', interpolate=False, legend=True)

plt.ylabel("Average Episodic Reward",fontsize=50)
plt.xlabel("Training Steps",fontsize=50)
plt.legend(prop={'size': 25})

plt.savefig(NAME)
plt.show()
