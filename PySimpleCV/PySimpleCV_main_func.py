import numpy as np 
import pandas as pd
from scipy import linalg
import re
from impedance import preprocessing
from impedance.models.circuits import Randles, CustomCircuit
import statsmodels.api as sm


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
    if CV_file.lower().endswith(".csv"):
        df_CV = pd.read_csv(CV_file,usecols=[0,1])
        df_CV = np.array(df_CV)
    elif CV_file.lower().endswith(".txt"):
        df_CV = pd.read_table(CV_file, sep='\t', header=None, usecols=[0,1])
        df_CV = np.array(df_CV)
    elif CV_file.lower().endswith(".par"):
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
    if bat_file.lower().endswith(".xls"):
        df_bat = pd.read_excel(bat_file,header=None)
        # Drop Index column, create our own
        df_bat = df_bat.drop([0],axis=1)
        
        # Delete all row that does not contain C_CC D_CC R
        row_size_raw_df_bat = len(df_bat)
        for i in range(0,row_size_raw_df_bat):
            bat_cell_state = pd.Series(df_bat[5])[i]
            if bat_cell_state != 'C_CC' and bat_cell_state != 'D_CC' and bat_cell_state != 'R' and bat_cell_state != 'C_CV' and bat_cell_state != 'D_CV':
                df_bat = df_bat.drop([i])
        df_bat.columns = ['time', 'volt', 'current', 'capacity', 'state']
        # Reset index after dropping some rows
        df_bat.reset_index(inplace=True)
        # convert '2-02:18:04' to seconds
        # Pandas datetime does not support changing format.
        row_size = len(df_bat)
        for i in range(0,row_size):
            df_bat['time'][i] = time2sec(df_bat['time'][i],'[:,-]')
            
        time_df = np.array(pd.Series(df_bat['time'])) #Get "time" column
        volt_df = np.array(pd.Series(df_bat['volt'].astype(float))) #Get "volt" column, convert to float
        current_df = np.array(pd.Series(df_bat['current'].astype(float))) #Get "Current" column, convert to float
        capacity_df = np.array(pd.Series(df_bat['capacity'].astype(float))) #Get "capacity" column, convert to float
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
    if list_start_end_key_idx == []:
        # If a certain state is not found, return None
         start_end_segment = []
    else:
        start_end_segment = np.array(list_start_end_key_idx)
        start_end_segment = np.split(start_end_segment, len(start_end_segment)/2)
        start_end_segment = np.stack(start_end_segment)
    return start_end_segment

def search_pattern(lst, pattern):
    indices = []
    for i in range(len(lst)):
        if lst[i:i+len(pattern)] == pattern:
            indices.append(i)
    return indices

def get_CV_init(df_CV, ir_compen):
    cv_size = df_CV.shape[0]
    volt = df_CV[:,0]
    current = df_CV[:,1]
    
    volt = volt[~np.isnan(volt)]
    current = current[~np.isnan(current)]
    
    # iR compensation
    volt = volt - current*ir_compen   
    return cv_size, volt, current

def trim_cv(volt, current, cut_val_s, cut_val_e):
    if cut_val_s == cut_val_e:
        cut_val_e = cut_val_s+2
    if cut_val_s > cut_val_e:
        save_cut_val = cut_val_s
        cut_val_s = cut_val_e
        cut_val_e = save_cut_val    
    volt_trim = volt[cut_val_s:cut_val_e]
    current_trim = current[cut_val_s:cut_val_e]
    return volt_trim, current_trim, cut_val_s, cut_val_e

def get_CV_peak(volt_trim, current_trim, cut_val_s, cut_val_e, peak_range, peak_pos, trough_pos, jpa_lns, jpa_lne, jpc_lns, jpc_lne, ir_compen):
    # Search for peak between peak_range.
        
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

    # if cut_val_s == cut_val_e:
    #     cut_val_e = cut_val_s+2
    # if cut_val_s > cut_val_e:
    #     save_cut_val = cut_val_s
    #     cut_val_s = cut_val_e
    #     cut_val_e = save_cut_val    
    
    # cv_size, volt, current = get_CV_init(df_CV, ir_compen)

    # volt_trim = volt[cut_val_s:cut_val_e]
    # current_trim = current[cut_val_s:cut_val_e]

    cv_size_trim = cut_val_e - cut_val_s
    high_range_peak = np.where((peak_pos+peak_range)>=(cv_size_trim-1),(cv_size_trim-1),peak_pos+peak_range)
    low_range_peak = np.where((peak_pos-peak_range)>=0,peak_pos-peak_range,0)
    peak_curr_range = current_trim[low_range_peak:high_range_peak]
    peak_curr = max(peak_curr_range)
    peak_idx = np.argmin(np.abs(peak_curr_range-peak_curr))
    peak_volt = volt_trim[low_range_peak:high_range_peak][peak_idx]

    high_range_trough = np.where((trough_pos+peak_range)>=(cv_size_trim-1),(cv_size_trim-1),trough_pos+peak_range)
    low_range_trough = np.where((trough_pos-peak_range)>=0,trough_pos-peak_range,0)
    trough_curr_range = current_trim[low_range_trough:high_range_trough]
    trough_curr = min(trough_curr_range)
    trough_idx = np.argmin(np.abs(trough_curr_range-trough_curr))
    trough_volt = volt_trim[low_range_trough:high_range_trough][trough_idx] 

    jpa_lnfit = np.polyfit(volt_trim[jpa_lns:jpa_lne],current_trim[jpa_lns:jpa_lne], 1)
    jpa_base = jpa_lnfit[0]*peak_volt + jpa_lnfit[1]

    jpc_lnfit = np.polyfit(volt_trim[jpc_lns:jpc_lne],current_trim[jpc_lns:jpc_lne], 1)
    jpc_base = jpc_lnfit[0]*trough_volt + jpc_lnfit[1]
 
    return low_range_peak, high_range_peak, peak_volt, peak_curr, low_range_trough, high_range_trough, trough_volt, trough_curr, jpa_lns,jpa_lne,jpc_lns,jpc_lne, volt_trim, current_trim, jpa_base, jpc_base


def cv_inflection(df_CV, ir_compen):
    #Separate CV into top part and bottom
    cv_size, volt, current = get_CV_init(df_CV, ir_compen)
    

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
    charge_CC_seq = find_seg_start_end(state_df,'C_CC')
    discharge_CC_seq = find_seg_start_end(state_df,'D_CC')
    charge_CV_seq = find_seg_start_end(state_df,'C_CV')
    discharge_CV_seq = find_seg_start_end(state_df,'D_CV')
    rest_seq = find_seg_start_end(state_df,'R')
    return charge_CC_seq, discharge_CC_seq, rest_seq, charge_CV_seq, discharge_CV_seq

def get_battery_eff(row_size, time_df, volt_df, current_df, capacity_df, state_df, charge_seq, discharge_seq):
    # Calculate the area of charge and discharge cycle and find VE,CE,EE for each cycle
    VE_lst = []
    CE_lst = []
    charge_cap_lst = []
    discharge_cap_lst = []
    cycle_end = min(np.shape(charge_seq)[0],np.shape(discharge_seq)[0]) #take the min amount of cycle between the charge and dis
    # cycle_start = 1
    for i in range(0,cycle_end):
        # Error if the cycle is not complete charge sequence more than discharge sequence
        time_seq_C = time_df[charge_seq[i][0]:charge_seq[i][1]+1]
        volt_seq_C = volt_df[charge_seq[i][0]:charge_seq[i][1]+1]
        current_seq_C = current_df[charge_seq[i][0]:charge_seq[i][1]+1]
        charge_cap_seq_C = capacity_df[charge_seq[i][0]:charge_seq[i][1]+1] 
        
        time_seq_D = time_df[discharge_seq[i][0]:discharge_seq[i][1]+1]
        volt_seq_D = volt_df[discharge_seq[i][0]:discharge_seq[i][1]+1]
        current_seq_D = current_df[discharge_seq[i][0]:discharge_seq[i][1]+1]
        dis_cap_seq_C = capacity_df[discharge_seq[i][0]:discharge_seq[i][1]+1]
        
        int_vt_C = np.trapz(volt_seq_C,time_seq_C)
        int_vt_D = np.trapz(volt_seq_D,time_seq_D)
        int_ct_C = np.trapz(current_seq_C,time_seq_C)
        # During discharge, current is negative, must make to positive
        int_ct_D = -(np.trapz(current_seq_D,time_seq_D))
        VE = int_vt_D/int_vt_C
        CE = int_ct_D/int_ct_C

        charge_cap_seq = np.array(charge_cap_seq_C)[-1]
        dis_cap_seq = np.array(dis_cap_seq_C)[-1]

        charge_cap_lst.append(charge_cap_seq)
        discharge_cap_lst.append(dis_cap_seq)
      
        VE_lst.append(VE)
        CE_lst.append(CE)
    VE_arr = np.array(VE_lst) * 100 # convert to %
    CE_arr = np.array(CE_lst) * 100
    EE_arr = (VE_arr/100 * CE_arr/100)*100
    # print(charge_cap_lst)
    charge_cap_arr = np.array(charge_cap_lst)
    discharge_cap_arr = np.array(discharge_cap_lst)
    return VE_arr, CE_arr, EE_arr, charge_cap_arr, discharge_cap_arr, cycle_end

def cy_idx_state_range(state_df, cycle_start, cycle_end, charge_seq, discharge_seq):
    # Get index for beginning and end of specify cycle
    # Take all start and end of the cycle chosen, select the first and last.
    # For plotting purpose
    cycle_index = np.stack((charge_seq[cycle_start:cycle_end], discharge_seq[cycle_start:cycle_end])) #no need to include rest
    cycle_idx_start = np.amin(cycle_index)
    cycle_idx_end = np.amax(cycle_index)
    cycle_idx_range = [cycle_idx_start, cycle_idx_end]
    return cycle_idx_range

def eis_read_file(eis_file,eis_choose_file_type):
    if eis_choose_file_type == 'CSV (.csv)':
        frequencies, z = preprocessing.readFile(eis_file)
    elif eis_choose_file_type == 'Gamry (.dta)':
        frequencies, z = preprocessing.readFile(eis_file,instrument='gamry')
    elif eis_choose_file_type == 'zplot (.z)':
        frequencies, z = preprocessing.readFile(eis_file,instrument='zplot')
    elif eis_choose_file_type == 'Versastudio (.par)':
        frequencies, z = preprocessing.readFile(eis_file,instrument='versastudio')
    elif eis_choose_file_type == 'biologic (.mpt)': #.mpt
        frequencies, z = preprocessing.readFile(eis_file,instrument='biologic')
    elif eis_choose_file_type == 'Autolab (.txt)':
        frequencies, z = preprocessing.readFile(eis_file,instrument='autolab')
    elif eis_choose_file_type == 'Parstat (.txt)':
        frequencies, z = preprocessing.readFile(eis_file,instrument='parstat')
    elif eis_choose_file_type == 'PowerSuite (.txt)':
        frequencies, z = preprocessing.readFile(eis_file,instrument='powersuite')
    elif eis_choose_file_type == 'CHInstruments':
        frequencies, z = preprocessing.readFile(eis_file,instrument='chinstruments')
    else:
        raise Exception("EIS file type not support")
    return frequencies, z

def eis_fit(eis_file,freqmin,freqmax,cir_scheme, rm_im_R, initial_guess, CPE_bool):
    if eis_file.lower().endswith('.csv'):
        frequencies, z = preprocessing.readFile(eis_file)
    elif eis_file.lower().endswith('.par'):
        frequencies, z = preprocessing.readFile(eis_file,instrument='versastudio')
    else:
        print("EIS file format not support")
        
    frequencies, z = preprocessing.cropFrequencies(frequencies, z, freqmin=freqmin, freqmax=freqmax)
    if rm_im_R == True:
        frequencies, z = preprocessing.ignoreBelowX(frequencies, z)
    circuit = Randles(initial_guess=initial_guess, CPE=CPE_bool)
    circuit.fit(frequencies, z)
    z_fit = circuit.predict(frequencies)
    return frequencies, z, z_fit, circuit

def lowess(x,y,frac):
    lowess = sm.nonparametric.lowess(y, x, frac=frac)
    smh_x = lowess[:, 0]
    smh_y = lowess[:, 1]
    return smh_x, smh_y

def diff(x,y):
    # diff = np.gradient(current,volt)
    # lowess = sm.nonparametric.lowess(diff, volt, frac=frac)
    # smh_diff_volt = lowess[:, 0]
    # smh_diff_curr = lowess[:, 1]
    diff = np.gradient(y,x)
    return x, diff