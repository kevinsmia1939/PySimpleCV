import os
# import math
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import re
import statsmodels.api as sm

def search_string_in_file(file_name, string_to_search):
    line_number = 0
    list_of_results = []
    with open(file_name, 'r') as read_file:
        for line in read_file:
            line_number += 1
            if string_to_search in line:
                list_of_results.append((line_number, line.rstrip()))
    return list_of_results

def CV_file2df(CV_file,cv_format):
    if cv_format == "CSV":
        df_CV = pd.read_csv(CV_file,usecols=[0,1])
        file_scan_rate = float(0)
        df_CV = np.array(df_CV)
    elif cv_format == "text":
        df_CV = pd.read_table(CV_file, sep='\t', header=None, usecols=[0,1])
        file_scan_rate = float(0)
        df_CV = np.array(df_CV)
    elif cv_format == "VersaSTAT":
        # Search for line match beginning and end of CV data and give ln number
        start_segment = search_string_in_file(CV_file, 'Definition=Segment')[0][0]
        end_segment = search_string_in_file(CV_file, '</Segment')[0][0]

        # Count file total line number
        with open(CV_file, 'r') as file:
            ln_count = sum(1 for _ in file)
        with open(CV_file, 'r') as file:
            # Search for scan rate value
            # Search for the pattern using regex
            match = re.search(r'Scan Rate \(V/s\)=([\d.]+)', file.read())
            if match:
                # Extract the value from the matched pattern
                file_scan_rate = float(match.group(1))
            else:
                file_scan_rate = float(0)
        footer = ln_count-end_segment
        df_CV = pd.read_csv(CV_file, skiprows=start_segment, skipfooter=footer, usecols=[2,3], header=None, engine='python')
    elif cv_format == "CorrWare":
        start_segment = search_string_in_file(CV_file, 'End Comments')[0][0]

        with open(CV_file, 'r') as file:
            # Search for scan rate value
            # Search for the pattern using regex
            match = re.search(r'Scan Rate:\s+(\d+)', file.read())
            if match:
                # Extract the value from the matched pattern
                file_scan_rate = float(match.group(1))
            else:
                file_scan_rate = float(0)    
        footer = 0
        df_CV = pd.read_csv(CV_file,sep='\t',skiprows=start_segment, skipfooter=footer, usecols=[0,1], header=None, engine='python')
    else:
        raise Exception("Unknown file type, please choose . cor, .csv, .par, .txt")
    # Remove NaN
    df_CV = df_CV.dropna()
    df_CV = np.array(df_CV)
    return df_CV, file_scan_rate

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

def open_battery_data(file_path,separate):
    # file_path string
    # separate string
    _, file_extension = os.path.splitext(file_path)
    if file_extension == '.xlsx' or file_extension == '.xls':
        df = pd.read_excel(file_path)
    elif file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension == '.txt':
        df = pd.read_csv(file_path, sep=separate, engine='python', header=None)
    elif file_extension == '.ods':
        df = pd.read_excel(file_path, engine='odf')
    else:
        print(f"Unsupported file format: {file_extension}. Please use .xlsx, .csv, .txt, .ods, or add a feature request")
        return None
    return df

def group_index(arr,key):
    arr = np.array(arr)
    state_ls = []
    for i in np.arange(0,len(arr),1):
      size = len(arr)
      if arr[i] == key:
        if i == 0 and arr[i+1] != key:
          state_ls.append(0,1)
        elif i == 0:
          state_ls.append(0)
        elif i > 0 and i < size-1:
          if arr[i-1] != key:
            state_ls.append(i)
          if arr[i+1] != key:
            state_ls.append(i+1)
        elif i == size-1:
            state_ls.append(i)
        elif i == size-1 and arr[i-1] != key:
          state_ls.append(size-1,size)
    state_group = np.array(state_ls).reshape(-1, 2)
    return state_group

def df_select_column(df,volt_col,current_col,time_col,rm_num_col):
    # Open all voltage, current, time header if not None
    df = df[[col for col in [volt_col,current_col,time_col] if col is not None]]
    if rm_num_col is not None:
    # Create a mask where True indicates the value is numeric
        df = df[df[rm_num_col].apply(lambda x: str(x).isnumeric())]
    df = df.reset_index(drop=True)
    return df    

def cut_list_to_shortest(a,b):
    min_length = min(len(a), len(b))
    a = a[:min_length]
    b = b[:min_length]
    return a,b

def calculate_battery(df,volt_col,current_col,time_col,rm_num_col,battery_cycle_select,voltage_name,current_name,time_name,charge_val,discharge_val,rest_val):
    try:
        charge_val = [*map(float, charge_val)]    
        charge_val = sorted(charge_val)
    except ValueError:
        charge_val = None
    try:
        discharge_val = [*map(float, discharge_val)]
        discharge_val = sorted(discharge_val)        
    except ValueError:
        discharge_val = None    
    try:
        rest_val = [*map(float, rest_val)]  
        rest_val = sorted(rest_val)
    except ValueError:
        rest_val = None    
        
    # df = df_select_column(df,volt_col,current_col,time_col,rm_num_col)

    state_list = []
    
    for i in np.arange(0,df.shape[0],1):
        curve = df.loc[i, battery_cycle_select]
        if charge_val != None and curve >= charge_val[0] and curve <= charge_val[1]:
                state_list.append("charge")
        elif discharge_val != None and curve >= discharge_val[0] and curve <= discharge_val[1]:
                state_list.append("discharge")
        elif rest_val != None and curve >= rest_val[0] and curve <= rest_val[1]:
                state_list.append("rest")
        else:
            state_list.append("null")  #Does not fit into any condition
    # print(state_list)
    state_col = np.array(state_list)
    # print(df.to_string())
    charge_group = group_index(state_list,"charge")
    discharge_group = group_index(state_list,"discharge")
    rest_group = group_index(state_list,"rest")
    null_group = group_index(state_list,"null")
    
    volt_area_charge_list = []
    volt_area_discharge_list = []
    index_time_size = 0.1869 # time between 2 index is 0.1869 s
    for k in charge_group:
        time = np.arange(k[0],k[1])*index_time_size
        volt_area_charge = np.trapz(df[voltage_name][k[0]:k[1]],time)
        volt_area_charge_list.append(volt_area_charge)
    for l in discharge_group:
        time = np.arange(l[0],l[1])*index_time_size
        volt_area_discharge = np.trapz(df[voltage_name][l[0]:l[1]],time)
        volt_area_discharge_list.append(volt_area_discharge)
    
    volt_area_discharge_list,volt_area_charge_list = cut_list_to_shortest(volt_area_discharge_list,volt_area_charge_list)
    VE = np.array(volt_area_discharge_list)/np.array(volt_area_charge_list)

    current_area_charge_list = []
    current_area_discharge_list = []
    for m in charge_group:
        time = np.arange(m[0],m[1])*index_time_size
        current_area_charge = np.trapz(df[current_name][m[0]:m[1]],time)
        current_area_charge_list.append(current_area_charge)
    for n in discharge_group:
        time = np.arange(n[0],n[1])*index_time_size
        current_area_discharge = np.trapz(df[current_name][n[0]:n[1]],time)
        current_area_discharge_list.append(current_area_discharge)

    current_area_discharge_list,current_area_charge_list = cut_list_to_shortest(current_area_discharge_list,current_area_charge_list)
    cap_discharge_arr = np.abs(np.array(current_area_discharge_list))
    cap_charge_arr = np.abs(np.array(current_area_charge_list))
    CE = np.abs(np.array(current_area_discharge_list))/np.abs(np.array(current_area_charge_list))
    EE = CE*VE
    return EE*100, VE*100, CE*100, cap_discharge_arr, cap_charge_arr, state_col

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

def get_CV_init(df_CV):
    volt = df_CV[:,0]
    current = df_CV[:,1]
    volt = volt[np.isfinite(volt)]
    current = current[np.isfinite(current)]
    cv_size = int(len(volt))
    return cv_size, volt, current

def ir_compen_func(volt,current,ir_compen):
    volt_compen = volt - current*ir_compen
    return volt_compen

def get_peak_CV(peak_mode,cv_size, volt, current, peak_range, peak_pos, jp_lns, jp_lne):
    # If peak range is given as 0, then peak is just where peak position is
    if peak_mode in ("exact","deflection"):
        peak_range = 0
        peak_curr = current[peak_pos]
        peak_volt = volt[peak_pos]   
        low_range_peak = peak_pos
        high_range_peak = peak_pos
    # Search for peak between peak_range.     
    else:
        high_range_peak = np.where((peak_pos+peak_range)>=(cv_size-1),(cv_size-1),peak_pos+peak_range)
        low_range_peak = np.where((peak_pos-peak_range)>=0,peak_pos-peak_range,0)
        peak_curr_range = current[low_range_peak:high_range_peak]
     
        if peak_mode == "max":
            peak_curr = max(peak_curr_range)
        elif peak_mode == "min":
            peak_curr = min(peak_curr_range)
            
        peak_idx = np.argmin(np.abs(peak_curr_range-peak_curr))     
        peak_volt = volt[low_range_peak:high_range_peak][peak_idx]
           
    # If the extrapolation coordinate overlapped, just give horizontal line
    if (volt[jp_lns:jp_lne]).size == 0:
        volt_jp = np.array([0, 1])
        current_jp = np.array([0, 0])
    else:
        volt_jp = volt[jp_lns:jp_lne]
        current_jp = current[jp_lns:jp_lne]
        
    jp_lnfit_coef,_ = poly.polyfit(volt_jp,current_jp, 1, full=True) # 1 for linear fit  
    jp_poly1d = poly.Polynomial(jp_lnfit_coef) 
    jp = peak_curr - jp_poly1d(peak_volt)
    return low_range_peak, high_range_peak, peak_volt, peak_curr, jp, jp_poly1d

def linear_fit(volt, current):
    fit_coef,_ = poly.polyfit(volt,current, 1, full=True) # 1 for linear fit  
    poly1d = poly.Polynomial(fit_coef) 
    return fit_coef, poly1d 

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

def lowess_func(x,y,frac):
    lowess = sm.nonparametric.lowess(y, x, frac=frac)
    smh_x = lowess[:, 0]
    smh_y = lowess[:, 1]
    return smh_x, smh_y

def diff(x,y):
    diff_y = np.gradient(y,x)
    # Find indices where x is not NaN
    valid_indices = np.isfinite(diff_y)
    # Use boolean indexing to select values in y corresponding to valid x values
    x = x[valid_indices]
    diff_y = diff_y[valid_indices]
    return x, diff_y #This is y

def lowess_diff(x_idx,x,y,frac):
    _, smh_y = lowess_func(x_idx,y,frac)
    x, smh_diff_y = diff(x,smh_y)
    return x, smh_diff_y

def peak_2nd_deriv(cv_size,volt,current):
    # The idx_arr is use to "unwarp" the circular CV
    idx_arr = np.arange(0,cv_size)
    frac = 0.05
    _,smh_curr = lowess_func(idx_arr,current,frac)
    _,smh_volt = lowess_func(idx_arr,volt,frac)
    smh_volt,diff1_curr = diff(smh_volt,smh_curr) #First diff, find peaks (slope = 0)
    idx_arr = np.arange(0,len(smh_volt)) # Recalculate size of idx_arr because NaN and inf removed
    smh_volt,diff2_curr = lowess_diff(idx_arr,smh_volt,diff1_curr,0.05)
    smh_volt,diff3_curr = lowess_diff(idx_arr,smh_volt,diff2_curr,0) #Detect deflection
    idx_intc_peak = idx_intercept(0,diff1_curr)
    idx_intc_defl = idx_intercept(0,diff3_curr)
    return idx_intc_peak, idx_intc_defl 

def idx_intercept(yint,y):
    idx_intc = []
    y = np.squeeze(y)
    for i in np.arange(1,y.size):
        if y[i] == yint:
            idx_intc.append(i)
        elif y[i] < yint and y[i-1] > yint or y[i] > yint and y[i-1] < yint: #in negative
            new_x = np.interp(yint, y[i-1:i+1], [i-1,i])
            idx_intc.append(new_x)
    return list(idx_intc)

def diffusion(scan,jp,alpha,conc_bulk,n):
# For more info - Electrochemical Methods: Fundamentals and Applications, 3rd Edition Allen J. Bard, Larry R. Faulkner, Henry S. White
# - Redox Flow Batteries: How to Determine Electrochemical Kinetic Parameters, Hao Wang et al.
# scan_rate_arr - scan rate,unit in volt
# jp - peak current density, unit in A/cm2
# alpha - charge-transfer coefficient, no unit
# conc_bulk - Bulk concentration, unit in mol/cm3
# n - number of electrons, no unit
    jp = np.abs(jp) 
    sqrt_scan = np.sqrt(scan)
    try: 
        jp_arr_lnfit, _ = poly.polyfit(sqrt_scan,jp,1,full=True)
        jp_arr_poly = poly.Polynomial(jp_arr_lnfit)
        jp_slope = jp_arr_lnfit[1] # take slope
        D_rev = (jp_slope/(2.69*(10**5)*n**(3/2)*conc_bulk))**2 # reversible
        D_irr = (jp_slope/(2.99*(10**5)*n**(3/2)*(alpha**0.5)*conc_bulk))**2 # irreversible  
    except SystemError:
        pass  
    # Calculate R2
    jp_fit = jp_arr_poly(sqrt_scan)
    residuals = jp - jp_fit
    ssr = np.sum(residuals ** 2)
    sst = np.sum((jp - np.mean(jp)) ** 2)
    r2 = (1 - (ssr / sst))
    return sqrt_scan, jp_fit ,D_irr ,D_rev ,r2

def reaction_rate(e_e0,jp,conc_bulk,n):
    jp = np.abs(jp)
    lnjp = np.log(jp)
    try:     
        lnjp_lnfit, _ = poly.polyfit(e_e0,lnjp,1,full=True)
        lnjp_poly = poly.Polynomial(lnjp_lnfit)
        lnjp_b = lnjp_lnfit[0] # take intercept
        slope = lnjp_lnfit[1]
        F = 96485.332
        alpha_cat = -slope*8.314472*298.15/F #cathodic where slope is negative
        alpha_ano = 1 + alpha_cat #anodic where slope is positive
        k0 = np.exp(lnjp_b-np.log(0.227*F*n*conc_bulk))
    except SystemError:
        pass
    # Calculate R2
    lnjp_fit = lnjp_poly(e_e0)       
    residuals = lnjp - lnjp_fit
    ssr = np.sum(residuals ** 2)
    sst = np.sum((lnjp - np.mean(lnjp)) ** 2)
    r2 = (1 - (ssr / sst))
    return lnjp, lnjp_fit, k0, alpha_cat, alpha_ano, r2

def find_alpha(volt,curr,jp_lns,peak_pos,jp_poly1d,jp,peak_volt):
    volt_eval_jp = volt[jp_lns:peak_pos]
    curr_eval_jp = curr[jp_lns:peak_pos]
    try: 
        baseline_eval_jp = np.linspace(jp_poly1d(volt[jp_lns]),jp_poly1d(volt[peak_pos]),volt_eval_jp.size)
        curr_baseline_jp = curr_eval_jp-baseline_eval_jp
        ep12_jp_idx = (np.abs(curr_baseline_jp-jp/2)).argmin()
        ep12_jp = volt_eval_jp[ep12_jp_idx] #Potential at peak current 1/2 (Ep 1/2)
        jp12_jp = curr_eval_jp[ep12_jp_idx]
        alpha_jp = 1-((47.7/1000)/np.abs(peak_volt - ep12_jp))
    except (ValueError, IndexError):
        ep12_jp = 0
        jp12_jp = 0
        alpha_jp = 0
    return ep12_jp, jp12_jp, alpha_jp

def convert_ref_elec():
    ref_she = 0
    ref_sce_sat = 0.241 #Saturated calomel electrode
    ref_cse = 0.314
    ref_agcl_sat = 0.197 # saturated
    ref_agcl_3molkg = 0.210 # 3 mol KCl/kg
    ref_agcl_3moll = 0. # 3.0 mol KCl/L
    ref_hg2so4_sat = 0.64 # saturated k2so4
    ref_hg2so4_05 = 0.68 # 0.5 M H2SO4
    return 0
    
def min_max_peak(peak_mode,cv_size, volt, current, peak_range, peak_pos):
    high_range_peak = np.where((peak_pos+peak_range)>=(cv_size-1),(cv_size-1),peak_pos+peak_range)
    low_range_peak = np.where((peak_pos-peak_range)>=0,peak_pos-peak_range,0)
    peak_curr_range = current[low_range_peak:high_range_peak]
    
    if peak_mode == 'max':
        peak_curr = max(peak_curr_range)
        peak_idx = np.argmin(np.abs(peak_curr_range-peak_curr))
        peak_volt = volt[low_range_peak:high_range_peak][peak_idx]
    elif peak_mode == 'min':
        peak_curr = min(peak_curr_range)
        peak_idx = np.argmin(np.abs(peak_curr_range-peak_curr))
        peak_volt = volt[low_range_peak:high_range_peak][peak_idx]
    elif peak_mode == 'none':
        peak_curr = current[peak_pos]
        peak_volt = volt[peak_pos]
    peak_real_idx = int(peak_pos-peak_range+peak_idx)
    return high_range_peak, low_range_peak, peak_volt, peak_curr, peak_real_idx

def check_val(val, val_type, err_val):
    if val_type == "int":
        try:
            value = int(val)
        except ValueError:
            value = int(err_val)
    elif val_type == "float":
        try:
            value = float(val)
        except ValueError:
            value = float(err_val)
    return value

def switch_val(a,b):
    if a >= b:
        b_old = b
        b = a
        a = b_old
    if a == b: #Prevent overlapped
        b = a+1
    return a,b

def RDE_kou_lev(ror,lim_curr,conc_bulk,n,kinvis,ror_unit_arr):
    unit_mapping = {'RPM': 0.104719755,'rad/s': 1}
    conv_unit_arr = [unit_mapping.get(item, item) for item in ror_unit_arr] #Convert RPM to rad/s
    ror = ror * conv_unit_arr
    inv_sqrt_ror = 1/np.sqrt(ror)
    inv_lim_curr = 1/lim_curr 
    try:
        if kinvis <= 0:
            kinvis = np.NaN
        j_inv_lnfit, _ = poly.polyfit(inv_sqrt_ror,inv_lim_curr,1,full=True)
        kou_lev_polyfit = poly.Polynomial(j_inv_lnfit)
        j_kin = 1/j_inv_lnfit[0]
        
        slope = 1/j_inv_lnfit[1]
        F = 96485.332 #Faraday constant
        # Levich equation
        diffusion = (slope/(0.62*n*F*kinvis**(-1/6)*conc_bulk))**(3/2) #cathodic where slope is negative
        # Calculate R2
        j_inv_fit = kou_lev_polyfit(inv_sqrt_ror)   
        residuals = inv_lim_curr - j_inv_fit
        ssr = np.sum(residuals ** 2)
        sst = np.sum((inv_lim_curr - np.mean(inv_lim_curr)) ** 2)
        r2 = (1 - (ssr / sst))
    except SystemError:
        j_kin=0
        r2=0
        j_inv_fit = []
        inv_sqrt_ror = []
        diffusion = np.NaN
        kou_lev_polyfit = np.NaN
    return inv_sqrt_ror, j_inv_fit, diffusion, j_kin, kou_lev_polyfit, r2

def data_poly_inter(x,y,poly_coef,detail):
    #Find the value of intersection
    # x,y is an array of equal length
    # poly_coef is 2nd data as numpy poly.Polynomial()
    #detail is number of points
    abs_min = np.abs(y-poly_coef(x))
    min_idx = np.argmin(abs_min)
    if min_idx == len(x):
        xvals1 = np.linspace(x[min_idx-1], x[min_idx],detail)
        yinterp1 = np.interp(xvals1, x, y)
        yinterp2 = np.interp(xvals1, x, poly_coef(x))
        fine_min_idx = np.argmin(np.abs(yinterp1-yinterp2))
    elif min_idx == 0:
        xvals1 = np.linspace(x[min_idx],x[min_idx+1],detail)
        yinterp1 = np.interp(xvals1, x, y)
        yinterp2 = np.interp(xvals1, x, poly_coef(x))
        fine_min_idx = np.argmin(np.abs(yinterp1-yinterp2))              
    elif abs_min[min_idx+1] > abs_min[min_idx-1]:
        xvals1 = np.linspace(x[min_idx-1], x[min_idx],detail)
        yinterp1 = np.interp(xvals1, x, y)
        yinterp2 = np.interp(xvals1, x, poly_coef(x))
        fine_min_idx = np.argmin(np.abs(yinterp1-yinterp2))
    elif abs_min[min_idx+1] < abs_min[min_idx-1]:
        xvals1 = np.linspace(x[min_idx],x[min_idx+1],detail)
        yinterp1 = np.interp(xvals1, x, y)
        yinterp2 = np.interp(xvals1, x, poly_coef(x))
        fine_min_idx = np.argmin(np.abs(yinterp1-yinterp2)) 
    elif abs_min[min_idx+1] == abs_min[min_idx-1]:
        fine_min_idx = min_idx
        xvals1 = x
        yinterp1 = y
    return xvals1[fine_min_idx], yinterp1[fine_min_idx]  