import numpy as np 
import pandas as pd
import re

def search_string_in_file(file_name, string_to_search):
    line_number = 0
    list_of_results = []
    with open(file_name, 'r') as read_obj:
        for line in read_obj:
            line_number += 1
            if string_to_search in line:
                list_of_results.append((line_number, line.rstrip()))
    return list_of_results

def CV_file2df(CV_file):
    if CV_file.endswith(".csv"):
        df_CV = pd.read_csv(CV_file,usecols=[0,1])
        df_CV = np.array(df_CV)
    elif CV_file.endswith(".txt"):
        df_CV = pd.read_table(CV_file, sep='\t', header=None, usecols=[0,1])
        df_CV = np.array(df_CV)
    elif CV_file.endswith(".par"):
        # Search for line match beginning and end of CV data and give ln number
        start_segment = search_string_in_file(CV_file, 'Definition=Segment')[0][0]
        end_segment = search_string_in_file(CV_file, '</Segment')[0][0]
        # Count file total line number
        with open(CV_file) as f:
            ln_count = sum(1 for _ in f)
        footer = ln_count-end_segment
        df_CV = pd.read_csv(CV_file,skiprows=start_segment, skipfooter=footer,usecols=[2,3],engine='python')
        df_CV = np.array(df_CV)
    else:
        raise Exception("Unknown file type, please choose .csv, .par")
    return df_CV

def battery_xls2df(bat_file):
    if bat_file.endswith(".xls"):
        df_bat = pd.read_excel(bat_file,header=None)
        # Drop Index column, create our own
        df_bat = df_bat.drop([0],axis=1)
        # Delete all row that does not contain C_CC D_CC R
        row_size_raw_df_bat = len(df_bat)
        for i in range(0,row_size_raw_df_bat):
            if pd.Series(df_bat[5])[i] != 'C_CC':
                if pd.Series(df_bat[5])[i] != 'D_CC':
                    if pd.Series(df_bat[5])[i] != 'R':
                            df_bat = df_bat.drop([i])
        # Reset index                    
        df_bat = df_bat.reset_index().drop(['index'], axis=1)
        df_bat.columns = ['time', 'volt', 'current', 'capacity', 'state']
        # convert '2-02:18:04' to seconds
        # Pandas datetime does not support changing format.
        row_size = len(df_bat)
        for i in range(0,row_size):
            df_bat['time'][i] = time2sec(df_bat['time'][i],'[:,-]')
            
        time_df = np.array(pd.Series(df_bat['time'])) #Get "time" column
        volt_df = np.array(pd.Series(df_bat['volt'])) #Get "volt" column
        current_df = pd.Series(df_bat['current']) #Get "Current" column
        capacity_df = pd.Series(df_bat['capacity']) #Get "capacity" column
        state_df = pd.Series(df_bat['state']) #Get "state" column
    else:
        raise Exception("Unknown file type, please choose .xls")
    return df_bat,row_size, time_df, volt_df, current_df, capacity_df, state_df

def find_seg_start_end(state_df,search_key):
    list_start_end_key_idx = []
    row_size = len(state_df)
    # if at the very beginning of state_df match our keyword,
    # then it start there.
    if state_df[0] == search_key:
       key_start = 0
       list_start_end_key_idx.append(key_start)
    for i in range(1,row_size):
        if state_df[i] == search_key:
            if state_df[i-1] != search_key:
                key_start = i
                list_start_end_key_idx.append(key_start)
        if state_df[i] != search_key:
            if state_df[i-1] == search_key:
                key_end = i-1
                list_start_end_key_idx.append(key_end)
    if state_df[row_size-1] == search_key:
       key_end = row_size-1
       list_start_end_key_idx.append(key_end)         
    start_end_segment = np.array(list_start_end_key_idx)
    start_end_segment = np.split(start_end_segment, len(start_end_segment)/2)
    start_end_segment = np.stack(start_end_segment)
    return start_end_segment

def get_CV_init(df_CV):
    cv_size = df_CV.shape[0]
    volt = df_CV[:,0]
    current = df_CV[:,1]
    return cv_size, volt, current

def max_value(set_value,input_val):
    # input_val would not be more than set_value
    if input_val >= set_value:
        max_output = set_value
    else:
        max_output = input_val
    return max_output

def min_value(set_value,input_val):
    # input_val would not be less than set_value
    if input_val <= set_value:
        min_output = set_value
    else:
        min_output = input_val
    return min_output

def get_CV_peak(df_CV, peak_range, peak_pos, trough_pos):
    # Search for peak between peak_range.
    cv_size, volt, current = get_CV_init(df_CV)  

    high_range_peak = np.where((peak_pos+peak_range)>=(cv_size-1),(cv_size-1),peak_pos+peak_range)
    low_range_peak = np.where((peak_pos-peak_range)>=0,peak_pos-peak_range,0)
    peak_curr_range = current[low_range_peak:high_range_peak]
    peak_curr = max(peak_curr_range)
    peak_idx = np.argmin(np.abs(peak_curr_range-peak_curr))
    peak_volt = volt[low_range_peak:high_range_peak][peak_idx]

    high_range_trough = np.where((trough_pos+peak_range)>=(cv_size-1),(cv_size-1),trough_pos+peak_range)
    low_range_trough = np.where((trough_pos-peak_range)>=0,trough_pos-peak_range,0)
    trough_curr_range = current[low_range_trough:high_range_trough]
    trough_curr = min(trough_curr_range)
    trough_idx = np.argmin(np.abs(trough_curr_range-trough_curr))
    trough_volt = volt[low_range_trough:high_range_trough][trough_idx]  
    return low_range_peak, high_range_peak, peak_volt, peak_curr, low_range_trough, high_range_trough, trough_volt, trough_curr

def get_CV(df_CV,jpa_lns,jpa_lne,jpc_lns,jpc_lne,peak_volt,trough_volt):
    # Select the points to extrapolate.
    if jpa_lns == jpa_lne:
        jpa_lne = jpa_lns+1
    if jpa_lns > jpa_lne:
        save_val_jpa = jpa_lns
        jpa_lns = jpa_lne
        jpa_lne = save_val_jpa
    if jpc_lns == jpc_lne:
        jpc_lne = jpc_lns+1
    if jpc_lns > jpc_lne:
        save_val_jpc = jpc_lns
        jpc_lns = jpc_lne
        jpc_lne = save_val_jpc
        
    cv_size, volt, current = get_CV_init(df_CV)    

    jpa_lnfit = np.polyfit(volt[jpa_lns:jpa_lne],current[jpa_lns:jpa_lne], 1)
    jpa_base = jpa_lnfit[0]*peak_volt + jpa_lnfit[1]

    jpc_lnfit = np.polyfit(volt[jpc_lns:jpc_lne],current[jpc_lns:jpc_lne], 1)
    jpc_base = jpc_lnfit[0]*trough_volt + jpc_lnfit[1]
    return jpa_lns,jpa_lne,jpc_lns,jpc_lne, volt, current, jpa_base, jpc_base

def time2sec(time_raw,delim):
    # Take time format such as 1-12:05:24 and convert to seconds
    time_raw = str(time_raw)
    time_sp = re.split(delim, time_raw)
    time_sp = list(map(int, time_sp))
    if len(time_sp) == 4:
        time_sec = time_sp[0]*3600*24 + time_sp[1]*3600 + time_sp[2]*60 + time_sp[3]
    elif len(time_sp) == 3:
        time_sec = time_sp[0]*3600 + time_sp[1]*60 + time_sp[2]
    return int(time_sec)

def find_state_seq(state_df):
    charge_seq = find_seg_start_end(state_df,'C_CC')
    discharge_seq = find_seg_start_end(state_df,'D_CC')
    rest_seq = find_seg_start_end(state_df,'R')
    return charge_seq, discharge_seq, rest_seq

def get_battery_eff(row_size, time_df, volt_df, current_df, capacity_df, state_df, cycle_start, cycle_end, charge_seq, discharge_seq):
    # Calculate the area of charge and discharge cycle and find VE,CE,EE for each cycle
    VE_lst = []
    CE_lst = []
    for i in range(cycle_start-1,cycle_end):
        time_seq_C_CC = time_df[charge_seq[i][0]:charge_seq[i][1]+1]
        volt_seq_C_CC = volt_df[charge_seq[i][0]:charge_seq[i][1]+1]
        current_seq_C_CC = current_df[charge_seq[i][0]:charge_seq[i][1]+1]
        time_seq_D_CC = time_df[discharge_seq[i][0]:discharge_seq[i][1]+1]
        volt_seq_D_CC = volt_df[discharge_seq[i][0]:discharge_seq[i][1]+1]
        current_seq_D_CC = current_df[discharge_seq[i][0]:discharge_seq[i][1]+1]
        int_vt_C = np.trapz(volt_seq_C_CC,time_seq_C_CC)
        int_vt_D = np.trapz(volt_seq_D_CC,time_seq_D_CC)
        int_ct_C = np.trapz(current_seq_C_CC,time_seq_C_CC)
        # During discharge, current is negative, must make to positive
        int_ct_D = -(np.trapz(current_seq_D_CC,time_seq_D_CC))
        VE = int_vt_D/int_vt_C
        CE = int_ct_D/int_ct_C
        VE_lst.append(VE)
        CE_lst.append(CE)
    VE_arr = np.array(VE_lst) * 100 # convert to %
    CE_arr = np.array(CE_lst) * 100
    EE_arr = (VE_arr/100 * CE_arr/100)*100
    return VE_arr, CE_arr, EE_arr

def cy_idx_state_range(state_df, cycle_start, cycle_end, charge_seq, discharge_seq):
    # Get index for beginning and end of specify cycle
    # Take all start and end of the cycle chosen, select the first and last.
    # For plotting purpose
    cycle_index = np.stack((charge_seq[cycle_start:cycle_end], discharge_seq[cycle_start:cycle_end])) #no need to include rest
    cycle_idx_start = np.amin(cycle_index)
    cycle_idx_end = np.amax(cycle_index)
    cycle_idx_range = [cycle_idx_start, cycle_idx_end]
    return cycle_idx_range