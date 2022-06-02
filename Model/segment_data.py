import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import view_as_windows


rootdir = '/Users/........../data/sample'


data_list = []
label_list = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(os.path.join(subdir, file))
        data = np.load(os.path.join(subdir, file), allow_pickle=True)
        ppg = view_as_windows(data[:, 4], (125*10), (125*10))
        #hr = np.mean(view_as_windows(data[:, 0], (125*15), (125*15)), axis=1)
        all_labels = np.mean(np.squeeze(view_as_windows(data[:, 0:3], (125*10, 3), (125*10))), axis=1)
        print(ppg.shape)
        print(all_labels.shape)
        data_list.append(ppg)
        label_list.append(all_labels)

data = np.concatenate(np.array(data_list, dtype=object), axis=0)
label = np.concatenate(np.array(label_list, dtype=object), axis=0)

np.save("data", data)
np.save("label", label)



#visualize signals
plt.figure(figsize=(12,4))
plt.xlabel("HR value [mmHg]")
plt.ylabel("Number of Windows")
#plt.ylim(0, 4)
plt.title("HR")
sns.distplot(label, bins=60, color='blue')
plt.show()


