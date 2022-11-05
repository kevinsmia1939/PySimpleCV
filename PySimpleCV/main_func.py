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
        df = pd.read_csv(CV_file,usecols=[0,1])
        df = np.array(df)
    elif CV_file.endswith(".txt"):
        df = pd.read_table(CV_file, sep='\t', header=None, usecols=[0,1])
        df = np.array(df)
    elif CV_file.endswith(".par"):
        # Search for line match beginning and end of CV data and give ln number
        start_segment = search_string_in_file('example_CV.par', 'Definition=Segment')[0][0]
        end_segment = search_string_in_file('example_CV.par', '</Segment')[0][0]
        # Count file total line number
        with open(CV_file) as f:
            ln_count = sum(1 for _ in f)
        footer = ln_count-end_segment
        df = pd.read_csv(CV_file,skiprows=start_segment, skipfooter=footer,usecols=[2,3],engine='python')
        df = np.array(df)
    else:
        raise Exception("Unknown file type, please choose .csv, .par")
    return df

def battery_xls2df(bat_file):
    if bat_file.endswith(".xls"):
        df = pd.read_excel(bat_file,header=None)
        # Drop Index column, create our own
        df = df.drop([0],axis=1)
        # Delete all row that does not contain C_CC D_CC R
        row_size_raw_df = len(df)
        for i in range(0,row_size_raw_df):
            if pd.Series(df[5])[i] != 'C_CC':
                if pd.Series(df[5])[i] != 'D_CC':
                    if pd.Series(df[5])[i] != 'R':
                            df = df.drop([i])
        # Reset index                    
        df = df.reset_index().drop(['index'], axis=1)
        df.columns = ['time', 'volt', 'current', 'capacity', 'state']
        # convert '2-02:18:04' to seconds
        # Pandas datetime does not support changing format.
        row_size = len(df)
        for i in range(0,row_size):
            df['time'][i] = time2sec(df['time'][i],'[:,-]')
            
        time_df = np.array(pd.Series(df['time'])) #Get "time" column
        volt_df = np.array(pd.Series(df['volt'])) #Get "volt" column
        current_df = pd.Series(df['current']) #Get "Current" column
        capacity_df = pd.Series(df['capacity']) #Get "capacity" column
        state_df = pd.Series(df['state']) #Get "state" column
    else:
        raise Exception("Unknown file type, please choose .xls")
    return df,row_size, time_df, volt_df, current_df, capacity_df, state_df

def find_seg_start_end(state_idx,search_seg):
    start_end_C_CC = []
    row_size = len(state_idx)
    if state_idx[0] == search_seg:
       start_C_CC = 0
       start_end_C_CC.append(start_C_CC)
    for i in range(1,row_size):
        if state_idx[i] == search_seg:
            if state_idx[i-1] != search_seg:
                start_C_CC = i
                start_end_C_CC.append(start_C_CC)
        if state_idx[i] != search_seg:
            if state_idx[i-1] == search_seg:
                end_C_CC = i-1
                start_end_C_CC.append(end_C_CC)
    if state_idx[row_size-1] == search_seg:
       end_C_CC = row_size-1
       start_end_C_CC.append(end_C_CC)         
    start_end_segment = np.array(start_end_C_CC)
    start_end_segment = np.split(start_end_segment, len(start_end_segment)/2)
    start_end_segment = np.stack(start_end_segment)
    return start_end_segment

def get_CV(CV_file,cut_val,jpa_lns,jpa_lne,jpc_lns,jpc_lne):
    df = CV_file2df(CV_file)
    
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

    volt = df[cut_val:,0] # exclude some index to remove artifacts
    current = df[cut_val:,1]
    
    jpa_lnfit = np.polyfit(volt[jpa_lns:jpa_lne],current[jpa_lns:jpa_lne], 1)
    idx_jpa_max = np.argmax(current)
    jpa_base = volt[idx_jpa_max]*jpa_lnfit[0]+jpa_lnfit[1]
    jpa_abs = current[idx_jpa_max]
    jpa = jpa_abs - jpa_base

    jpa_ref_ln = np.linspace(volt[jpa_lns],volt[idx_jpa_max],100)
    jpa_ref = jpa_ref_ln*jpa_lnfit[0]+jpa_lnfit[1]

    jpc_lnfit = np.polyfit(volt[jpc_lns:jpc_lne],current[jpc_lns:jpc_lne], 1)
    idx_jpc_min = np.argmin(current) #Find index of jpc peak
    jpc_base = volt[idx_jpc_min]*jpc_lnfit[0]+jpc_lnfit[1]
    jpc_abs = current[idx_jpc_min]
    jpc = jpc_base - jpc_abs

    jpc_ref_ln = np.linspace(volt[jpc_lns],volt[idx_jpc_min],100)
    jpc_ref = jpc_ref_ln*jpc_lnfit[0]+jpc_lnfit[1]
    return volt, current, jpa_ref_ln, jpa_ref, idx_jpa_max, jpa_abs, jpa_base, jpc_ref_ln, jpc_ref, idx_jpc_min, jpc_abs, jpc_base, jpa_lns, jpa_lne, jpc_lns, jpc_lne, jpa, jpc

def time2sec(time_raw,delim):
    time_raw = str(time_raw)
    time_sp = re.split(delim, time_raw)
    time_sp = list(map(int, time_sp))
    if len(time_sp) == 4:
        time_sec = time_sp[0]*3600*24 + time_sp[1]*3600 + time_sp[2]*60 + time_sp[3]
    elif len(time_sp) == 3:
        time_sec = time_sp[0]*3600 + time_sp[1]*60 + time_sp[2]
    return int(time_sec)