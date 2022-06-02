#import libraries
import io
import os
import pandas as pd
import numpy as np
import wfdb
from IPython.display import display
import matplotlib.pyplot as plt


#analyzing model by extracting a single patient record
signals, fields = wfdb.rdsamp('mimic3wdb/30/3000063/3000063_0006',
                              pn_dir='mimic3wdb/30/3000063',
                                     channel_names = ["PLETH", "ABP"]) #, sampfrom=0, sampto=7500)

wfdb.plot_items(signal=signals, fs=fields['fs'], time_units='samples',
                sig_units=['NU', 'mmHg'], figsize=(10,4), title= "Record patient 3000063")

print("Printing Signals")
display(signals)
print("Printing fields")
display(fields)
df_temp = pd.DataFrame(signals, columns=["PLETH", "ABP"])

#analyzing PPG and ABP signals
plt.figure(figsize=(12,4))
plt.xlabel("samples")
plt.ylabel("Amplitude [V]")
#plt.ylim(0, 4)
plt.title("PPG signal")
plt.plot(df_temp['PLETH'])
plt.show()

plt.figure(figsize=(12,4))
plt.xlabel("samples")
plt.ylabel("BP [mmHg]")
#plt.ylim(40, 140)
plt.title("ABP signal")
plt.plot(df_temp['ABP'])
plt.show()
