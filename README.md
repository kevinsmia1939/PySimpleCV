# PySimpleCV
Graphical user interface for plotting the cyclic voltammogram and calculating jpa, jpc, and reversibility.
PySimpleCV also plots battery cycles and calculates efficiencies. Written in pure Python.

Currently supports VersaStudio (.par),.csv, and.txt.
For.csv and .txt, the first column must be voltage and the second column is current.
Battery cycling support .xls files with state (C_CC, D_CC, R) columns
Support copy-paste battery efficiency.

Requires Python 3.10 and above.
Modules used: Numpy, Matplotlib, PySimpleGUI, and pandas

License: GPLv3

Future plans.
* More file format support
* More flexible state recognition (C_CC, D_CC, R, etc).
* Export plot.

The file PySimpleCV/PySimpleCV contain the main code responsible to produce GUI.
PySimpleCV/PySimpleCV_main_func.py contain mathematical function for calculation.

Installation
Clone this repository.
`git clone git@github.com:kevinsmia1939/PySimpleCV.git`
Navigate to the PySimpleCV folder and run the PySimpleCV executable. The PySimpleCV file must be marked as executable.

Use Flatpak for Linux.
`flatpak install flathub io.github.kevinsmia1939.PySimpleCV`

Uninstall
`flatpak remove io.github.kevinsmia1939.PySimpleCV`

![PySimpleCV](https://github.com/kevinsmia1939/PySimpleCV/blob/main/data/screenshot/cv_screenshot.png?raw=true)
![PySimpleCV](https://github.com/kevinsmia1939/PySimpleCV/blob/main/data/screenshot/battery_screenshot.png?raw=true)
