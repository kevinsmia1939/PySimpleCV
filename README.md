# PySimpleCV
Graphical user interface for plotting the cyclic voltammogram and calculating jpa, jpc, and reversibility.
PySimpleCV also plots battery cycles and calculates efficiencies. Written in pure Python.

Currently supports VersaStudio (.par),.csv, and.txt.
For.csv and .txt, the first column must be voltage and the second column is current.
Battery cycling support .xls files with state (C_CC, D_CC, R) columns
Support copy-paste battery efficiency.

Modules used: Numpy, Matplotlib, PySimpleGUI, and pandas

License: GPLv3

Future plan
* More file format support
* More flexible state recognition (C_CC, D_CC, R, etc).
* Export plot.

![PySimpleCV](https://codeberg.org/Andy_Great/PySimpleCV/raw/branch/main/data/screenshot/cv_screenshot.png?raw=true)
![PySimpleCV](https://codeberg.org/Andy_Great/PySimpleCV/raw/branch/main/data/screenshot/battery_screenshot.png?raw=true)
