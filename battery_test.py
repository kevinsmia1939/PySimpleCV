import pandas as pd
import numpy as np
import scipy.integrate as integrate
from main_func import find_seg_start_end, time2sec, battery_xls2df
import matplotlib.pyplot as plt
file_name = 'example_battery_charge-discharge_cycle_full.xls'

df, row_size, time_df, volt_df, current_df, capacity_df, state_df = battery_xls2df(file_name)

# plt.figure(figsize=(15,10))
# plt.plot(time_df, volt_df)
# plt.show()

charge_seq = find_seg_start_end(state_df,'C_CC')
discharge_seq = find_seg_start_end(state_df,'D_CC')
rest_seq = find_seg_start_end(state_df,'R')
# tot_cycle_number = np.shape(charge_seq)[0]

a = 6
cycle_start = a
cycle_end = a
cycle_index = np.stack((charge_seq[cycle_start-1:cycle_end], discharge_seq[cycle_start-1:cycle_end]))
# print(charge_seq[cycle_start-1:cycle_end])
# print(discharge_seq[cycle_start-1:cycle_end])
# print(rest_seq[cycle_start-1:cycle_end])

print(cycle_index)
cycle_idx_start = np.amin(cycle_index)
cycle_idx_end = np.amax(cycle_index)
# print(cycle_idx_start,cycle_idx_end)

# Calculate the area of charge and discharge cycle and find VE for each cycle
# VE_lst = []
# for i in range(cycle_start-1,cycle_end):
#     time_seq_C_CC = time_df[charge_seq[i][0]:charge_seq[i][1]+1]
#     volt_seq_C_CC = volt_df[charge_seq[i][0]:charge_seq[i][1]+1]
#     time_seq_D_CC = time_df[discharge_seq[i][0]:discharge_seq[i][1]+1]
#     volt_seq_D_CC = volt_df[discharge_seq[i][0]:discharge_seq[i][1]+1]
#     int_vt_C = integrate.trapezoid(volt_seq_C_CC,time_seq_C_CC)
#     int_vt_D = integrate.trapezoid(volt_seq_D_CC,time_seq_D_CC)
#     VE = int_vt_D/int_vt_C
#     VE_lst.append(VE)

# VE_arr = np.array(VE_lst)
# VE_avg = np.average(VE_arr)
# print(VE_arr)
# print(VE_avg)

# plt.figure(figsize=(15,10))
plt.plot(time_df[cycle_idx_start:cycle_idx_end], volt_df[cycle_idx_start:cycle_idx_end])

x = 411
y = 489+1
print(time_df[x:y])
plt.plot(time_df[x:y], volt_df[x:y],'--')
plt.show()