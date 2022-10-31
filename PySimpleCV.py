#!/usr/bin/python3
import numpy as np 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
from matplotlib.figure import Figure
import PySimpleGUI as sg 
import matplotlib 
import pandas as pd
matplotlib.use('TkAgg')

# CV_file = 'example_CV_data.txt'
# df=pd.read_table(CV_file, skiprows=1, sep='\t', header=None, usecols=[0,1])
# df = np.array(df)
# cv_size = df.shape[0]

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
        # sg.popup(df, keep_on_top=True)
    elif CV_file.endswith(".txt"):
        df = pd.read_table(CV_file, sep='\t', header=None, usecols=[0,1])
        df = np.array(df)
        # sg.popup(df, keep_on_top=True)
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
        # sg.popup(pd_line_count, keep_on_top=True)
    else:
        raise Exception("Unknown file type, please choose .csv, .par")
    return df

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
    [sg.Button('Exit'),sg.Button('About PySimpleCV')]
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
        case "Open CV File":
            CV_file = sg.popup_get_file('Choose CV file')
            # If cancel, close the window, go back to beginning
            if CV_file is None:
                continue
            elif CV_file == '':
                continue
            try:
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
