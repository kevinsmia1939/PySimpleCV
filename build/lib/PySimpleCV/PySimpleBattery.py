#!/usr/bin/python3
import numpy as np 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
from matplotlib.figure import Figure
import PySimpleGUI as sg 
import matplotlib
import scipy.integrate as integrate
# import pandas as pd
matplotlib.use('TkAgg')

from main_func import find_seg_start_end, time2sec, battery_xls2df

def get_Battery_init(bat_file):
    df, row_size, time_df, volt_df, current_df, capacity_df, state_df = battery_xls2df(bat_file)
    return df, row_size, time_df, volt_df, current_df, capacity_df, state_df

def find_state_seg(state_df):
    charge_seq = find_seg_start_end(state_df,'C_CC')
    discharge_seq = find_seg_start_end(state_df,'D_CC')
    rest_seq = find_seg_start_end(state_df,'R')
    return charge_seq, discharge_seq, rest_seq

def cy_idx_state_end(state_df, cycle_start, cycle_end, charge_seq, discharge_seq):
    cycle_index = np.stack((charge_seq[cycle_start-1:cycle_end], discharge_seq[cycle_start-1:cycle_end])) #no need to include rest
    cycle_idx_start = np.amin(cycle_index)
    cycle_idx_end = np.amax(cycle_index)+1
    return cycle_idx_start, cycle_idx_end

def get_bat_eff(df, row_size, time_df, volt_df, current_df, capacity_df, state_df, cycle_start, cycle_end, charge_seq, discharge_seq):
    # Calculate the area of charge and discharge cycle and find VE for each cycle
    VE_lst = []
    CE_lst = []
    for i in range(cycle_start-1,cycle_end):
        time_seq_C_CC = time_df[charge_seq[i][0]:charge_seq[i][1]+1]
        volt_seq_C_CC = volt_df[charge_seq[i][0]:charge_seq[i][1]+1]
        current_seq_C_CC = current_df[charge_seq[i][0]:charge_seq[i][1]+1]
        time_seq_D_CC = time_df[discharge_seq[i][0]:discharge_seq[i][1]+1]
        volt_seq_D_CC = volt_df[discharge_seq[i][0]:discharge_seq[i][1]+1]
        current_seq_D_CC = current_df[discharge_seq[i][0]:discharge_seq[i][1]+1]
        int_vt_C = integrate.trapezoid(volt_seq_C_CC,time_seq_C_CC)
        int_vt_D = integrate.trapezoid(volt_seq_D_CC,time_seq_D_CC)
        int_ct_C = integrate.trapezoid(current_seq_C_CC,time_seq_C_CC)
        # During discharge, current is negative, must make to positive
        int_ct_D = -(integrate.trapezoid(current_seq_D_CC,time_seq_D_CC))
        VE = int_vt_D/int_vt_C
        CE = int_ct_D/int_ct_C
        VE_lst.append(VE)
        CE_lst.append(CE)
    VE_arr = np.array(VE_lst)
    CE_arr = np.array(CE_lst)
    return VE_arr, CE_arr

def draw_figure(canvas, figure): 
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas) 
    figure_canvas_agg.draw() 
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1) 
    return figure_canvas_agg

def About_PySimpleBattery():
    [sg.popup('PySimpleBattery written by Kavin Teenakul',
    'License: GPL v3',
    'https://github.com/kevinsmia1939/PySimpleCV',
    'Libraries used: Numpy, Pandas, Matplotlib, PySimpleGUI')]

layout = [
    [sg.Canvas(key='-CANVAS-')],
    [sg.Button("Open Battery File"),sg.Text('.xls files')],
    [sg.Text('Battery file:'), sg.Text('No Battery file selected', key = 'bat_file_use')],
    [sg.Text('Average voltage efficiency='), sg.Text('', key = 'output_ve'),sg.Text('%')],
    [sg.Text('Average current efficiency='), sg.Text('', key = 'output_ce'),sg.Text('%')],
    [sg.Text('Average energy efficiency='), sg.Text('', key = 'output_ee'),sg.Text('%')],
    [sg.Text('Cycle start'), sg.Slider(range=(1, 1), size=(60, 10), orientation='h', key='cycle_start', enable_events=True, disabled=True)],
    [sg.Text('Cycle end'), sg.Slider(range=(1, 1), size=(60, 10), orientation='h', key='cycle_end', enable_events=True, disabled=True)],
    [sg.Button('Clear plot'),sg.Button('Exit'),sg.Button('About PySimpleBattery')]
    ]

window = sg.Window('PySimpleBattery', layout, finalize=True, element_justification='center')

canvas = window['-CANVAS-'].tk_canvas

fig = Figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax2.set_ylabel("Current")
ax1.set_xlabel("Time")
ax1.set_ylabel("Voltage")
ax1.grid()
fig_agg = draw_figure(canvas, fig)

while True:
    event, values = window.read()
    match event:
        case sg.WIN_CLOSED | "Exit":
            break
        case "About PySimpleBattery":
            About_PySimpleBattery()
        case "Clear plot":
            # ax1.cla()
            ax2.cla()
            ax2.clear() 
            ax1.cla()
            fig_agg.draw()
        case "Open Battery File":
            bat_file_new = sg.popup_get_file('Choose battery cycle file (.xls)')
            # If cancel, close the window, go back to beginning
            # If empty, use old file
            if bat_file_new is None:
                continue
            elif bat_file_new == '':
                continue
            try:
                bat_file = bat_file_new
                df, row_size, time_df, volt_df, current_df, capacity_df, state_df = battery_xls2df(bat_file)
                # Sequence information
                charge_seq, discharge_seq, rest_seq = find_state_seg(state_df)
                tot_cycle_number = np.shape(charge_seq)[0]
                VE_arr, CE_arr = get_bat_eff(df, row_size, time_df, volt_df, current_df, capacity_df, state_df, 1, tot_cycle_number, charge_seq, discharge_seq)
                window['cycle_start'].Update(range=(1,tot_cycle_number))
                window['cycle_end'].Update(range=(1,tot_cycle_number))
                # cycle_start will be enabled when cycle_end change, cycle_start cannot exceed cycle_end
                window['cycle_end'].Update(disabled=False)
                window['bat_file_use'].Update(bat_file)
                VE_avg = np.average(VE_arr)
                CE_avg = np.average(CE_arr)
                EE_avg = VE_avg * CE_avg
                window['output_ve'].Update(np.round(VE_avg*100,3))
                window['output_ce'].Update(np.round(CE_avg*100,3))
                window['output_ee'].Update(np.round(EE_avg*100,3))
                ax1.plot(time_df, volt_df, '-',color='blue')
                ax2.plot(time_df, current_df, '--',color='red')
                ax1.grid()
                fig_agg.draw()
            except Exception as file_error:
                sg.popup(file_error, keep_on_top=True)
        case 'cycle_start' | 'cycle_end':
            # ax2 = ax1.twinx() #This should be here
            # # Get value of cycle end
            cycle_end = int(values['cycle_end'])
            # Update start range equal to end
            window['cycle_start'].Update(range=(1,cycle_end))
            # Enable start slider
            window['cycle_start'].Update(disabled=False)
            cycle_start = int(values['cycle_start'])
            ax1.cla()
            ax1.grid()
            # Start plotting
            df, row_size, time_df, volt_df, current_df, capacity_df, state_df = battery_xls2df(bat_file)
            charge_seq, discharge_seq, rest_seq = find_state_seg(state_df)
            cycle_idx_start, cycle_idx_end = cy_idx_state_end(state_df, cycle_start, cycle_end, charge_seq, discharge_seq)
            VE_avg = np.average(VE_arr[cycle_start-1:cycle_end])
            CE_avg = np.average(CE_arr[cycle_start-1:cycle_end])
            EE_avg = VE_avg * CE_avg
            
            ax1.plot(time_df, volt_df, color='blue')
            left_bound = time_df[cycle_idx_start:cycle_idx_end][0]
            right_bound = time_df[cycle_idx_start:cycle_idx_end][-1]
            ax1.set_xlim(left=left_bound,right=right_bound)
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Voltage")
    
            ax2.cla()
            ax2.plot(time_df, current_df,'--',color='red')
            ax2.set_ylabel("Current")
            
            window['output_ve'].Update(np.round(VE_avg*100,3))
            window['output_ce'].Update(np.round(CE_avg*100,3))
            window['output_ee'].Update(np.round(EE_avg*100,3))
            fig_agg.draw()
window.close()
