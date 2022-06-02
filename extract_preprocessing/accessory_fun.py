
import os
import numpy as np
import urllib.request
import re
from IPython.display import display

# ========================================
# Accessory functions
# ========================================


def url_is_alive(url):
    """
    Checks that a given URL is reachable.
    :param url: A URL
    :rtype: bool
    """
    request = urllib.request.Request(url)
    request.get_method = lambda: 'HEAD'

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False


def list_available_records_for_fieldId(url_path:str,fileId:int)-> list:
    """
    reads patient records from the url for a given file series
    """
    wdb_path_toAllRecords = os.path.join(url_path,str(fileId),'RECORDS')
    #print(wdb_path_toAllRecords)
    if url_is_alive(wdb_path_toAllRecords):
        with urllib.request.urlopen(wdb_path_toAllRecords) as response:
            wdb_records = response.readlines()
        return [re.findall(r'\d+',str(record))[0] for record in wdb_records]
    else:
        raise ValueError('fileId {} doesnt exist in the mimic database'.format(fileId))


def check_header(header, waveform_signals):
    return set(waveform_signals).issubset(set(header.sig_name)) if header else False


def make_signal_lenght_eq(num, wav):
    if num.shape[0] < wav.shape[0]:
        wav = wav[0:num.shape[0], :]
    else:
        num = num[0:wav.shape[0], :]
    return num, wav


def fill_record(record, stacked_record):
    array = np.empty((stacked_record.shape[0]))
    array[:] = int(record)
    return np.column_stack((stacked_record, array))


def fill_empty_signal(signals, fields, numerics_signal, suffix='numeric'):
    if suffix == 'numeric':
        fields_present = fields['sig_name']
        missing_columns = set(numerics_signal) - set(fields_present)
        remaining_columns_data = np.empty((fields['sig_len'], len(missing_columns)))
        remaining_columns_data[:] = np.nan
        stacked_array = np.hstack((signals, remaining_columns_data))
    return stacked_array


def get_multi_segment_record(wdb_path_toAllRecords: str, record_idx: int):
    path = os.path.join(wdb_path_toAllRecords, str(record_idx), 'RECORDS')
    print(path)
    with urllib.request.urlopen(path) as response:
        multi_segment_records = response.readlines()
        return [str(record.decode("utf-8")).rstrip() for record in multi_segment_records]


def load_files(path: str) -> np.array:
    signals, fields = wfdb.rdsamp(temp_n, pn_dir=pn_dir_path)


def read_file_interpolate(signal_name, pn_dir_path, temp, header):
    print(signal_name, header.sig_name)
    if set(signal_name).issubset(set(header.sig_name)):
        print("True")
        signals, fields = wfdb.rdsamp(temp, pn_dir=pn_dir_path, channel_names=signal_name)
        interpolated_df = pd.DataFrame(signals).interpolate()
        return interpolated_df


def get_numerics_upsampled(signals, signals_n, fields, fields_n, numerics_signal):
    if fields_n and fields:
        signal_lenght = signals.shape[0]
        waveform_frequency = fields['fs']
        signal_n_lenght = signals_n.shape[0]
        numeric_frequency = fields_n['fs']

        # appending missing columns
        if not signals_n.shape[1] == 4:
            signals_n = fill_empty_signal(signals=signals_n, fields=fields_n, numerics_signal=numerics_signal,
                                          suffix='numeric')
            assert signals_n.shape[1] == 4

        # calculating sampling ratio/np.repeat(factor)
        signal_lenght_ratio = math.floor(signal_lenght / signal_n_lenght)
        repeat_frequency = fields['fs'] * 60 if math.ceil(1 / fields_n['fs']) == 60 else fields['fs'] * 1
        print(repeat_frequency, signal_lenght_ratio)

        numerics_upsampled = np.repeat(signals_n, repeat_frequency, axis=0)
        return numerics_upsampled


def get_stacked_record(numerics_upsampled, signals):
    stacked_record = np.hstack((numerics_upsampled, signals))
    print("stacked record for record id {} with total lenght {} and columns {}".format(record, stacked_record.shape[0],
                                                                                       stacked_record.shape[1]))
    assert stacked_record.shape[1] == 6
    stacked_record = fill_record(record=record, stacked_record=stacked_record)
    return stacked_record


def make_numerics_label_case_insensitive(header, signal_list):
    numerics_signal_lower = [i.lower() for i in numerics_signal]
    channel_name = [i for i in header.sig_name if i.lower() in numerics_signal_lower]
    return channel_name