import io
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import posixpath
import wfdb
import datetime
import urllib.request
import requests


# Open file with list of all records through response
DIR = 30
wdb_path_toAllRecords = 'https://archive.physionet.org/physiobank/database/mimic3wdb/' + str(DIR) + '/RECORDS'

with urllib.request.urlopen(wdb_path_toAllRecords) as response:
    wdb_records = response.readlines()
    print(wdb_records)


def extactnumber(lis):
    temp = []
    for btes in lis:
        btes = str(btes,'utf−8')
        temp.append(int("".join(filter(str.isdigit, btes))))
    return temp

#to list all the records in directory30
PatientList = extactnumber(wdb_records)
print(len(PatientList))
print(PatientList)


Plist = []
for rec in PatientList:
    id = str(DIR) + "/" + str(rec)
    print(id)

    url = 'https://archive.physionet.org/physiobank/database/mimic3wdb/' + id + '/RECORDS'

    with urllib.request.urlopen(url) as response:
        wdb_path_toPatientlist = response.readlines()

        for lines in wdb_path_toPatientlist:
            multi_seg = lines.decode("utf−8")
            multi_seg_rec = str(multi_seg).rstrip()
            Plist.append(multi_seg_rec)


print(Plist)
Patient_all_rec = Plist

signals_rec = [x for x in Patient_all_rec if "_" in x]
numerics_rec = [x for x in Patient_all_rec if "n" in x]
print(signals_rec)
print(numerics_rec)


df_temp1 = []
df_temp2 = []

df_main1 = []
df_main2 = []

x_num = ['HR', ('ABP Sys' and 'ABP SYS') , ('ABP Dias' and 'ABP DIAS'), ('ABP Mean' and 'ABP MEAN')]
x_sig = ['PLETH', 'ABP']


i=0
j=0


for labels in numerics_rec:
    label_rec = labels[:labels.index('n')]

    for record in signals_rec:
        record_rec = record[:record.index('_')]

        num_pn_dir_path = ('mimic3wdb/' + str(DIR) + "/" + label_rec)
        num_header = wfdb.rdheader(labels, pn_dir = num_pn_dir_path)

        sig_pn_dir_path = ('mimic3wdb/' + str(DIR) + "/" + record_rec)
        sig_header = wfdb.rdheader(record, pn_dir = sig_pn_dir_path)

        #display(header.__dict__)
        if (label_rec == record_rec):
            if "ABP" not in num_header.sig_name:
                continue

            if (all(x in sig_header.sig_name for x in ["PLETH", "ABP"]) & sig_header.fs == 1):
                print(sig_header.record_name, sig_header.sig_name, sig_header.sig_len)


            sig_temp = ('mimic3wdb/' + str(DIR) + "/" + record_rec
                       + "/" + sig_header.record_name)

            wave_signals, wave_fields = wfdb.rdsamp(sig_temp,
            pn_dir= sig_pn_dir_path, channel_names = ['PLETH', 'ABP'])
            wfdb.plot_items(signal=signals, fs=fields['fs'])
            display(signals)
            display(fields)

            numeric_signals, numeric_fields = wfdb.rdsamp('labels',pn_dir= num_pn_dir_path ,channel_names =['HR',
            ('ABP Sys' and 'ABP SYS'), ('ABP Dias' and 'ABP DIAS'),
            ('ABP Mean' and 'ABP MEAN')])


            df_temp1 = pd.DataFrame(wave_signals, columns= x_sig)
            df_temp2 = pd.DataFrame(numeric_signals, columns= x_num)

            #add datarame index
            df_main1.append(df_temp1)
            df_main2.append(df_temp2)

            df_temp1.drop
            df_temp2.drop

    i = i + 1
    print(i)


    if (i > 10):
        break

df_patient_sig = pd.concat(df_main1)
df_patient_num = pd.concat(df_main2)

df_patient_rec = df_patient_sig.to_csv('dir_patient_id.csv')
df_patient_label = df_patient_num.to_csv('dir_patient_label.csv')