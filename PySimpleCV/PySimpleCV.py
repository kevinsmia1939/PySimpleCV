#!/usr/bin/python3
import numpy as np 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
from matplotlib.figure import Figure
import PySimpleGUI as sg 
import matplotlib 
import scipy.integrate as integrate
# import pandas as pd
matplotlib.use('TkAgg')

from PySimpleCV_main_func import CV_file2df, get_CV, battery_xls2df, find_seg_start_end

def get_CV_init(CV_file):
    df = CV_file2df(CV_file)
    cv_size = df.shape[0]
    volt = df[:,0]
    current = df[:,1]
    return cv_size, volt, current

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

def About_PySimpleCV():
    [sg.popup('PySimpleCV written by Kavin Teenakul',
    'License: GPL v3',
    'https://github.com/kevinsmia1939/PySimpleCV',
    'Libraries used: Numpy, Pandas, Matplotlib, PySimpleGUI')]

cv_layout = [
    [sg.Canvas(key='-CANVAS_cv-')],
    [sg.Button("Open CV File"),sg.Text('.csv file with voltage on 1st colume and current on 2nd column separated by comma')],
    [sg.Text('CV file:'), sg.Text('No CV file selected', key = 'CV_file_use')],
    [sg.Text('jpa='), sg.Text('', key = 'output_jpa'), sg.Text('jpc='), sg.Text('', key = 'output_jpc'), sg.Text('Reversibility (jpa/jpc)='), sg.Text('', key = 'output_rev')],
    [sg.Text('Trim CV data'), sg.Slider(range=(1, 1), size=(60, 10), orientation='h', key='sl_cut_val', enable_events=True, disabled=True)],
    [sg.Text('Start 1'),sg.Slider(range=(1, 1), size=(60, 10), orientation='h', key='sl_jpa_lns', enable_events=True, disabled=True)],
    [sg.Text('End 1'),sg.Slider(range=(1, 1), size=(60, 10), orientation='h', key='sl_jpa_lne', enable_events=True, disabled=True)],
    [sg.Text('Start 2'),sg.Slider(range=(1, 1), size=(60, 10), orientation='h', key='sl_jpc_lns', enable_events=True, disabled=True)],
    [sg.Text('End 2'),sg.Slider(range=(1, 1), size=(60, 10), orientation='h', key='sl_jpc_lne', enable_events=True, disabled=True)],
    [sg.Button('Clear plot', key='cv_clear'),sg.Button('Exit', key='exit'),sg.Button('About PySimpleCV')]
]

bat_layout = [
    [sg.Canvas(key='-CANVAS_bat-')],
    [sg.Button("Open Battery File"),sg.Text('.xls files')],
    [sg.Text('Battery file:'), sg.Text('No Battery file selected', key = 'bat_file_use')],
    [sg.Text('Average voltage efficiency='), sg.Text('', key = 'output_ve'),sg.Text('%')],
    [sg.Text('Average current efficiency='), sg.Text('', key = 'output_ce'),sg.Text('%')],
    [sg.Text('Average energy efficiency='), sg.Text('', key = 'output_ee'),sg.Text('%')],
    [sg.Text('Cycle start'), sg.Slider(range=(1, 1), size=(60, 10), orientation='h', key='cycle_start', enable_events=True, disabled=True)],
    [sg.Text('Cycle end'), sg.Slider(range=(1, 1), size=(60, 10), orientation='h', key='cycle_end', enable_events=True, disabled=True)],
    [sg.Button('Clear plot',key='bat_clear'),sg.Button('Exit', key='exit_2')]
    ]

layout =[[sg.TabGroup([[  sg.Tab('Cyclic Voltammetry', cv_layout),
                           sg.Tab('Battery Cycling', bat_layout)
                           ]], key='-TAB GROUP-', expand_x=True, expand_y=True),
]]
          
window = sg.Window('PySimpleCV', layout, finalize=True, element_justification='center')
# canvas = window['-CANVAS-'].tk_canvas


canvas_cv = window['-CANVAS_cv-'].tk_canvas
canvas_bat = window['-CANVAS_bat-'].tk_canvas

fig_cv = Figure()
fig_bat = Figure()
ax_cv = fig_cv.add_subplot(111)
ax_bat_volt = fig_bat.add_subplot(111)
ax_cv.set_xlabel("Voltage")
ax_cv.set_ylabel("Current")
ax_cv.grid()

ax_bat_current = ax_bat_volt.twinx()
ax_bat_current.set_ylabel("Current")
ax_bat_volt.set_xlabel("Time")
ax_bat_volt.set_ylabel("Voltage")
ax_bat_volt.grid()
fig_agg_cv = draw_figure(canvas_cv, fig_cv)
fig_agg_bat = draw_figure(canvas_bat, fig_bat)

while True:
    event, values = window.read()
    match event:
        case sg.WIN_CLOSED | 'exit' | 'exit_2':
            break
        case "About PySimpleCV":
            About_PySimpleCV()
        case 'cv_clear':
            ax_cv.cla()
            fig_agg_cv.draw()
        case "Open CV File":
            CV_file_new = sg.popup_get_file('Choose CV file')
            # If cancel, close the window, go back to beginning
            if CV_file_new is None:
                continue
            elif CV_file_new == '':
                continue
            try:
                CV_file = CV_file_new
                cv_size, volt, current = get_CV_init(CV_file) # Only need the cv_size so set to [0]
                window['sl_cut_val'].Update(range=(0,cv_size))
                window['sl_jpa_lns'].Update(range=(0,cv_size))
                window['sl_jpa_lne'].Update(range=(0,cv_size))
                window['sl_jpc_lns'].Update(range=(0,cv_size))
                window['sl_jpc_lne'].Update(range=(0,cv_size))
                window['sl_cut_val'].Update(disabled=False)
                window['sl_jpa_lns'].Update(disabled=False)
                window['sl_jpa_lne'].Update(disabled=False)
                window['sl_jpc_lns'].Update(disabled=False)
                window['sl_jpc_lne'].Update(disabled=False)
                window['CV_file_use'].Update(CV_file)
                ax_cv.grid()
                ax_cv.plot(volt, current, '-')
                ax_cv.grid()
                fig_agg_cv.draw()
            except Exception as file_error:
                sg.popup(file_error, keep_on_top=True)
        case 'sl_cut_val' | 'sl_jpa_lns' | 'sl_jpa_lne' | 'sl_jpc_lns' | 'sl_jpc_lne' :
            cut_val = int(values['sl_cut_val'])  # Getting the k value from the slider element.
            jpa_lns = int(values['sl_jpa_lns'])
            jpa_lne = int(values['sl_jpa_lne'])
            jpc_lns = int(values['sl_jpc_lns'])
            jpc_lne = int(values['sl_jpc_lne'])
            ax_cv.cla()
            ax_cv.grid()
            # Start plotting
            volt, current, jpa_ref_ln, jpa_ref, idx_jpa_max, jpa_abs, jpa_base, jpc_ref_ln, jpc_ref, idx_jpc_min, jpc_abs, jpc_base, jpa_lns, jpa_lne, jpc_lns, jpc_lne, jpa, jpc = get_CV(CV_file,cut_val,jpa_lns,jpa_lne,jpc_lns,jpc_lne)
            ax_cv.plot(volt, current)
            ax_cv.set_xlabel("Voltage")
            ax_cv.set_ylabel("Current")
            ax_cv.plot(volt[jpa_lns:jpa_lne],current[jpa_lns:jpa_lne],linewidth=4,linestyle='-',color='red')
            ax_cv.plot(jpa_ref_ln,jpa_ref,linewidth=2,linestyle='--')
            ax_cv.plot(volt[idx_jpa_max],jpa_abs,'bo')
            ax_cv.plot(volt[idx_jpa_max],jpa_base,'go')
            ax_cv.annotate(text='', xy=(volt[idx_jpa_max],jpa_base), xytext=(volt[idx_jpa_max],jpa_abs), arrowprops=dict(arrowstyle='<-'))
            
            ax_cv.plot(volt[jpc_lns:jpc_lne],current[jpc_lns:jpc_lne],linewidth=4,linestyle='-',color='blue')
            ax_cv.plot(jpc_ref_ln,jpc_ref,linewidth=2,linestyle='--')
            ax_cv.plot(volt[idx_jpc_min],jpc_abs,'bo')
            ax_cv.plot(volt[idx_jpc_min],jpc_base,'go')
            ax_cv.annotate(text='', xy=(volt[idx_jpc_min],jpc_abs), xytext=(volt[idx_jpc_min],jpc_base), arrowprops=dict(arrowstyle='<-'))
            window['output_jpa'].Update(np.round(jpa,3))
            window['output_jpc'].Update(np.round(jpc,3))
            window['output_rev'].Update(np.round(jpa/jpc,3))
            fig_agg_cv.draw()
        case 'bat_clear':
            ax_bat_current.cla()
            ax_bat_volt.cla()
            fig_agg_bat.draw()
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
                ax_bat_volt.plot(time_df, volt_df, '-',color='blue')
                ax_bat_current.plot(time_df, current_df, '--',color='red')
                ax_bat_volt.grid()
                fig_agg_bat.draw()
            except Exception as file_error:
                sg.popup(file_error, keep_on_top=True)
        case 'cycle_start' | 'cycle_end':
            # ax2 = ax1.twinx() #This should not be here
            # # Get value of cycle end
            cycle_end = int(values['cycle_end'])
            # Update start range equal to end
            window['cycle_start'].Update(range=(1,cycle_end))
            # Enable start slider
            window['cycle_start'].Update(disabled=False)
            cycle_start = int(values['cycle_start'])
            ax_bat_volt.cla()
            ax_bat_volt.grid()
            # Start plotting
            df, row_size, time_df, volt_df, current_df, capacity_df, state_df = battery_xls2df(bat_file)
            charge_seq, discharge_seq, rest_seq = find_state_seg(state_df)
            cycle_idx_start, cycle_idx_end = cy_idx_state_end(state_df, cycle_start, cycle_end, charge_seq, discharge_seq)
            VE_avg = np.average(VE_arr[cycle_start-1:cycle_end])
            CE_avg = np.average(CE_arr[cycle_start-1:cycle_end])
            EE_avg = VE_avg * CE_avg
            
            ax_bat_volt.plot(time_df, volt_df, color='blue')
            left_bound = time_df[cycle_idx_start:cycle_idx_end][0]
            right_bound = time_df[cycle_idx_start:cycle_idx_end][-1]
            ax_bat_volt.set_xlim(left=left_bound,right=right_bound)
            ax_bat_volt.set_xlabel("Time")
            ax_bat_volt.set_ylabel("Voltage")
    
            ax_bat_current.cla()
            ax_bat_current.plot(time_df, current_df,'--',color='red')
            ax_bat_current.set_ylabel("Current")
            
            window['output_ve'].Update(np.round(VE_avg*100,3))
            window['output_ce'].Update(np.round(CE_avg*100,3))
            window['output_ee'].Update(np.round(EE_avg*100,3))
            fig_agg_bat.draw()
window.close()
