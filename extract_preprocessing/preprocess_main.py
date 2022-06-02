from extract_preprocessing.accessory_fun import *
from extract_preprocessing.functions import *


# ========================================
# Main run script
# ========================================

import time



start = time.time()

url = 'https://archive.physionet.org/physiobank/database/mimic3wdb'
fileId = 30
numerics_signal = ['HR', 'ABP SYS', 'ABP DIAS', 'ABP MEAN']
waveform_signals = ["PLETH", "ABP"]

records = list_available_records_for_fieldId(url_path= url, fileId= fileId)

final_list = []

#for record in ["3000393"]:
for record in records[0:10]:#records[2:8]:
    fieldId = record[0:2]
    multi_sec_records = get_multi_segment_record(url + os.sep + fieldId, record)
    pn_dir_path = "mimic3wdb" + os.sep + str(fieldId) + os.sep + str(record) + os.sep

    try:
        header = wfdb.rdheader(record + '_layout', pn_dir=pn_dir_path)
    except:
        print("layout file absent,skip this record {}".format(record))
        continue

    header_n = wfdb.rdheader(record + 'n', pn_dir=pn_dir_path)
    channel_names = make_numerics_label_case_insensitive(header_n, numerics_signal)
    header_val = check_header(header, waveform_signals)
    temp_path_n = pn_dir_path + str(record) + 'n'
    temp_path = pn_dir_path + str(record)

    if (str(record) + 'n' in multi_sec_records) and (str(record) in multi_sec_records) and header_val:
        print("processing records for recording {}".format(record))
        header_n = wfdb.rdheader(record + 'n', pn_dir=pn_dir_path)
        channel_names = make_numerics_label_case_insensitive(header_n, numerics_signal)
        try:
            print("reading numerics file")
            signals_n, fields_n = wfdb.rdsamp(temp_path_n, pn_dir=pn_dir_path, channel_names=channel_names)
        except:
            print("record {} doesn't have numerics data present".format(record))

        try:
            print("reading waveform files")

            #             wfdb.plot_items(signal=record.p_signal,
            #                            figsize=(10,4), ecg_grids='all')
            signals, fields = wfdb.rdsamp(temp_path, pn_dir=pn_dir_path,
                                          channel_names=waveform_signals)  # ,sampfrom=0, sampto=100000)
        except:
            print("record {} doesn't have waveform data present".format(record))

        if fields_n and fields:
            numerics_upsampled = get_numerics_upsampled(signals, signals_n, fields, fields_n, numerics_signal)
            if (numerics_upsampled.shape[0] == signals.shape[0]):
                print("equal lenght datasets and now stacking")
                stacked_record = get_stacked_record(numerics_upsampled, signals)
            else:
                print("shapes not equal numerics lenght is :{} and waveform lenght is :{}".format(
                    numerics_upsampled.shape[0], signals.shape[0]))
                print("making the shape equal and then concatenating")
                numerics_upsampled, signals = make_signal_lenght_eq(numerics_upsampled, signals)
                assert numerics_upsampled.shape[0] == signals.shape[0]
                stacked_record = get_stacked_record(numerics_upsampled, signals)

            df = pd.DataFrame(stacked_record, columns=numerics_signal + waveform_signals + ['record'])
            # interpolating
            df1 = df.interpolate()  # fill meaningful values instead of NA's
            df1 = df1.iloc[:, 0:5]

            # signal segmentation creating windows
            time_len = 30  # in sec
            freq = 125
            window_shape = (time_len * freq)

            ids_without_na_for_all_cols = [
                get_window_ids_without_na(convert_data_into_batches(df1[i].values, window_shape)) \
                for i in df1.columns]


            id_bp_vals = get_ids_of_reliable_bp_values(convert_data_into_batches(df1['HR'].values, window_shape),\
                                                       convert_data_into_batches (df1['ABP SYS'].values, window_shape),\
                                                       convert_data_into_batches(df1['ABP DIAS'].values, window_shape), \
                                                       [50, 200], [55, 185], [35, 120], 3750)

            id_flat_lines = remove_flat_lines(convert_data_into_batches(df1['PLETH'].values, window_shape),
                                              repeat_val=15)

            intersection_ids_na = ids_without_na_for_all_cols[0].intersection, \
                (ids_without_na_for_all_cols[1], ids_without_na_for_all_cols[2], \
                 ids_without_na_for_all_cols[3], ids_without_na_for_all_cols[4])

            # selecting common window ids with non_na rows and right bp values
            intersection_ids = intersection_ids_na.intersection(id_bp_vals, id_flat_lines)

            fixed_window_df = pd.DataFrame(
                {i: subset_data(df1[i].values, window_shape, intersection_ids) for i in df1.columns})


            #remove noise
            fixed_window_df['PLETH'] = butter_bandpass_filter(fixed_window_df['PLETH'])  # remove noise

            id_peaks = get_ids_of_windows_with_non_peaks(
                convert_data_into_batches(fixed_window_df['PLETH'], window_shape), prominence_val=2.8)

            fixed_window_df_after_find_peaks = pd.DataFrame(
                {i: subset_data(fixed_window_df[i].values, window_shape, id_peaks) for i in fixed_window_df.columns})


            # scaling

            fixed_window_df_after_find_peaks['PLETH'] = scaling_arr(fixed_window_df_after_find_peaks['PLETH'].values)

            # save it with your local path here
            fixed_window_df_after_find_peaks.to_csv("/Users/nikitha/mimic/" + str(record) + ".csv")

            stacked = np.stack(fixed_window_df_after_find_peaks.values)
            np.save(f"preprocessed_data", stacked)


    else:
        print("one of the values for record {} is false among recordn: {},record: {}, header.sig_name: {} {}".format(
            record, str(record) + 'n' in multi_sec_records, str(record) in multi_sec_records, header_val,
            header.sig_name))

end = time.time()
print("total time taken to read the records, time: {}".format(end - start))


