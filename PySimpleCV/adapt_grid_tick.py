import PySimpleGUI as sg
import numpy as np
import math

layout = [
    [sg.Button('Plot and Zoom')],
    [sg.Graph(canvas_size=(500, 500), graph_bottom_left=(-1, -1), graph_top_right=(1, 1), key='graph', background_color='white', enable_events=True)],
]

window = sg.Window('Plot', layout, finalize=True)
x = np.arange(-10,10, 0.1)
y = np.sin(10*x)*np.sin(0.5*x)
# data1 = list(zip(x, y))

window['graph'].erase()
graph_elem = window['graph']

def grid_tick(number,updown):
    if number == 0:
        return 0
    elif number != 0 and updown == "up":
        magnitude = 10 ** (int(math.log10(abs(number))) - 1)
        rounded_number = math.ceil(number / magnitude) * magnitude
    elif number != 0 and updown == "down":
        magnitude = 10 ** (int(math.log10(abs(number))) - 1)
        rounded_number = math.floor(number / magnitude) * magnitude
    elif number != 0 and updown == None:
        magnitude = 10 ** (int(math.log10(abs(number))) - 1)
        rounded_number = round(number / magnitude) * magnitude    
    return rounded_number
k = 1
while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    if event == 'Plot and Zoom':
        window['graph'].erase()
        
        # Create 5% margin around the whole data
        x_margin = (max(x)-min(x))*0.05
        y_margin = (max(y)-min(y))*0.05
        canvas_lef = min(x)-x_margin
        canvas_rig = max(x)+x_margin
        canvas_bot = min(y)-y_margin
        canvas_top = max(y)+y_margin        
        graph_elem.change_coordinates((canvas_lef,canvas_bot), (canvas_rig,canvas_top))
        
        tick_loc = 0.92
        tick_len = 0.03
        for i in np.linspace(min(x),max(x),10):
            print(i)
            xtick_id = graph_elem.draw_line((i,(canvas_bot*tick_loc)-tick_len), (i,(canvas_bot*tick_loc)+tick_len))
            xtext_id = graph_elem.draw_text(round(i,2),(i,(canvas_bot*tick_loc)-tick_len*2), color='green')     
        
        # x_range = 10/k
        # y_range = x_range #Square plot     
        
        x_axis_ln_id = graph_elem.draw_line((min(x), canvas_bot*tick_loc), (max(x), canvas_bot*tick_loc)) #x  
        graph_elem.draw_lines(list(zip(x, y)), color='blue')
        
        # # x_ticks = np.arange(0,200,1)
        # # test = 2**n
        # # test2 = 1**n
        # graph_size = x_range + x_range
        # for x in np.arange(-20,20,1):
        #     # print(x)
        #     xtick_id = graph_elem.draw_line((x,-y_range*0.05), (x,y_range*0.05))
        #     xtext_id = graph_elem.draw_text(x,(x,-4), color='green')        
        
        # k += 1
        # print(k)
        # x_axis_ln_id = cv_graph.draw_line(graph_bottomleft, graph_topright) #x
window.close()
