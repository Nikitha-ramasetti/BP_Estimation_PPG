
import pandas as pd

# ========================================
# Extract_Preprocessing
# ========================================


def skip_windows_with_flat_lines(non_nan_windows, repeat_val):
    arr = []
    for i in non_nan_windows:
        flag = False
        for item, count in Counter(i).items():
            if count > repeat_val:
                flag = True
                break
        if flag == True:
            continue
        else:
            arr.append(i)
    return arr


def scaling_arr(arr):
    arr = np.reshape(arr, (arr.shape[0], -1))
    scaler = MinMaxScaler()
    scaler.fit(arr)
    scaled_array = scaler.transform(arr)
    return scaled_array


def fill_nan(A):
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interp1d(inds[good], A[good], bounds_error=False)
    B = np.where(np.isfinite(A), A, f(inds))
    return B


def butter_bandpass(lowcut=0.5, highcut=8, fs=125, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut=0.5, highcut=8, fs=125, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def subset_data(arr, window_shape, intersection_ids):
    x = view_as_windows(arr, window_shape)
    x_sub = x[list(intersection_ids)]
    return x_sub


def get_window_ids_without_na(arr: pd.Series):
    return set([idx for idx, data in enumerate(arr) if np.all((~np.isnan(data)))])


def convert_data_into_batches(arr, window_size):
    batch = math.ceil(arr.shape[0] / window_size)
    start_id = 0
    batches = []
    for i in range(batch):
        if i < batch - 1:
            batch_size = arr[start_id:window_size + i * (window_size)]
            batches.append(batch_size)
            start_id = start_id + window_size
        elif i == (batch - 1):
            batch_size = arr[start_id:arr.shape[0]]
            batches.append(batch_size)
    return batches


def remove_flat_lines(arr_2d, repeat_val):
    """
    returns id of windows with flat lines from the signal. Flat lines are detected based on whether a \
    single value exceeds threshold repeat_val specified in the argument. Higher the repeat_val less, \
    less will be occurence of such data and so more data will be returned
    """
    idx = []
    for k, i in enumerate(arr_2d):
        unique, counts = np.unique(i, return_counts=True)
        if np.all(counts < repeat_val):
            idx.append(k)
    return set(idx)


def get_ids_of_windows_with_non_peaks(arr_2d, prominence_val):
    idx = []
    for k, i in enumerate(arr_2d):
        peaks, _ = find_peaks(i, height=0, prominence=prominence_val)
        if len(peaks) == 0:
            idx.append(k)
    return set(idx)


def get_ids_of_reliable_bp_values(arr_hr,arr_sys,arr_dias,
                                  hr_limit:list,sys_limit:list,dias_limit:list, window_shape):
    x_hr_idx = [k for k,i in enumerate(arr_hr) if np.all(i > hr_limit[0]) and np.all(i < hr_limit[1])]
    x_sys_idx = [k for k,i in enumerate(arr_sys) if np.all(i > sys_limit[0]) and np.all(i < sys_limit[1])]
    x_dias_idx = [k for k,i in enumerate(arr_dias) if np.all(i > dias_limit[0]) and np.all(i < dias_limit[1])]
    x_idx = set(x_hr_idx).intersection(set(x_sys_idx)).intersection(set(x_dias_idx))
    return x_idx


def subset_data(arr, window_shape, intersection_ids):
    data_list = []
    data = convert_data_into_batches(arr, window_shape)
    for id in intersection_ids:
        data_list.append(data[id])
    return (np.hstack(data_list))
