#!/usr/bin/python3
import numpy as np 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
from matplotlib.figure import Figure
import PySimpleGUI as sg 
import matplotlib 
# import pandas as pd
matplotlib.use('TkAgg')

from PYSimpleCV_main_func import CV_file2df, get_CV

def get_CV_init(CV_file):
    df = CV_file2df(CV_file)
    cv_size = df.shape[0]
    volt = df[:,0]
    current = df[:,1]
    return cv_size, volt, current

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

layout = [
    [sg.Canvas(key='-CANVAS-')],
    [sg.Button("Open CV File"),sg.Text('.csv file with voltage on 1st colume and current on 2nd column separated by comma')],
    [sg.Text('CV file:'), sg.Text('No CV file selected', key = 'CV_file_use')],
    [sg.Text('jpa='), sg.Text('', key = 'output_jpa'), sg.Text('jpc='), sg.Text('', key = 'output_jpc'), sg.Text('Reversibility (jpa/jpc)='), sg.Text('', key = 'output_rev')],
    [sg.Text('Trim CV data'), sg.Slider(range=(1, 1), size=(60, 10), orientation='h', key='sl_cut_val', enable_events=True, disabled=True)],
    [sg.Text('Start 1'),sg.Slider(range=(1, 1), size=(60, 10), orientation='h', key='sl_jpa_lns', enable_events=True, disabled=True)],
    [sg.Text('End 1'),sg.Slider(range=(1, 1), size=(60, 10), orientation='h', key='sl_jpa_lne', enable_events=True, disabled=True)],
    [sg.Text('Start 2'),sg.Slider(range=(1, 1), size=(60, 10), orientation='h', key='sl_jpc_lns', enable_events=True, disabled=True)],
    [sg.Text('End 2'),sg.Slider(range=(1, 1), size=(60, 10), orientation='h', key='sl_jpc_lne', enable_events=True, disabled=True)],
    [sg.Button('Clear plot'),sg.Button('Exit'),sg.Button('About PySimpleCV')]
]

window = sg.Window('PySimpleCV', layout, finalize=True, element_justification='center')

canvas = window['-CANVAS-'].tk_canvas

fig = Figure()
ax = fig.add_subplot(111)
ax.set_xlabel("Voltage")
ax.set_ylabel("Current")
ax.grid()
fig_agg = draw_figure(canvas, fig)

while True:
    event, values = window.read()
    match event:
        case sg.WIN_CLOSED | "Exit":
            break
        case "About PySimpleCV":
            About_PySimpleCV()
        case "Clear plot":
            ax.cla()
            fig_agg.draw()
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
                ax.grid()
                ax.plot(volt, current, '-')
                ax.grid()
                fig_agg.draw()
            except Exception as file_error:
                sg.popup(file_error, keep_on_top=True)
        case 'sl_cut_val' | 'sl_jpa_lns' | 'sl_jpa_lne' | 'sl_jpc_lns' | 'sl_jpc_lne' :
            cut_val = int(values['sl_cut_val'])  # Getting the k value from the slider element.
            jpa_lns = int(values['sl_jpa_lns'])
            jpa_lne = int(values['sl_jpa_lne'])
            jpc_lns = int(values['sl_jpc_lns'])
            jpc_lne = int(values['sl_jpc_lne'])
            ax.cla()
            ax.grid()
            # Start plotting
            volt, current, jpa_ref_ln, jpa_ref, idx_jpa_max, jpa_abs, jpa_base, jpc_ref_ln, jpc_ref, idx_jpc_min, jpc_abs, jpc_base, jpa_lns, jpa_lne, jpc_lns, jpc_lne, jpa, jpc = get_CV(CV_file,cut_val,jpa_lns,jpa_lne,jpc_lns,jpc_lne)
            ax.plot(volt, current)
            ax.set_xlabel("Voltage")
            ax.set_ylabel("Current")
            ax.plot(volt[jpa_lns:jpa_lne],current[jpa_lns:jpa_lne],linewidth=4,linestyle='-',color='red')
            ax.plot(jpa_ref_ln,jpa_ref,linewidth=2,linestyle='--')
            ax.plot(volt[idx_jpa_max],jpa_abs,'bo')
            ax.plot(volt[idx_jpa_max],jpa_base,'go')
            ax.annotate(text='', xy=(volt[idx_jpa_max],jpa_base), xytext=(volt[idx_jpa_max],jpa_abs), arrowprops=dict(arrowstyle='<-'))
            
            ax.plot(volt[jpc_lns:jpc_lne],current[jpc_lns:jpc_lne],linewidth=4,linestyle='-',color='blue')
            ax.plot(jpc_ref_ln,jpc_ref,linewidth=2,linestyle='--')
            ax.plot(volt[idx_jpc_min],jpc_abs,'bo')
            ax.plot(volt[idx_jpc_min],jpc_base,'go')
            ax.annotate(text='', xy=(volt[idx_jpc_min],jpc_abs), xytext=(volt[idx_jpc_min],jpc_base), arrowprops=dict(arrowstyle='<-'))
            window['output_jpa'].Update(np.round(jpa,3))
            window['output_jpc'].Update(np.round(jpc,3))
            window['output_rev'].Update(np.round(jpa/jpc,3))
            fig_agg.draw()
window.close()
