# PySimpleCV
Graphical user interface for plotting the cyclic voltammogram and calculating jpa, jpc, and reversibility.
PySimpleCV also plots battery cycles and calculates efficiencies. Written in pure Python.

Currently supports VersaStudio (.par),.csv, and.txt.
For.csv and .txt, the first column must be voltage and the second column is current.
Battery cycling support .xls files with state (C_CC, D_CC, R) columns
Support copy-paste battery efficiency.

Modules used: Numpy, Matplotlib, PySimpleGUI, and pandas

License: GPLv3

Future plans.
* More file format support
* More flexible state recognition (C_CC, D_CC, R, etc).
* Export plot.

The file PySimpleCV/PySimpleCV contain the main code responsible to produce GUI.
PySimpleCV/PySimpleCV_main_func.py contain mathematical function for calculation.

![PySimpleCV](https://github.com/kevinsmia1939/PySimpleCV/blob/main/data/screenshot/cv_screenshot.png?raw=true)
![PySimpleCV](https://github.com/kevinsmia1939/PySimpleCV/blob/main/data/screenshot/battery_screenshot.png?raw=true)
