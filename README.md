# PySimpleCV
Graphical user interface for plotting the cyclic voltammogram and calculating jpa, jpc, and reversibility.
PySimpleCV also plots battery cycles and calculates efficiencies. Written in pure Python.

Please cite: https://doi.org/10.5281/zenodo.8019091

**Cyclic voltammetry**

Currently supports VersaStudio (.par),.csv, and.txt.
For.csv and .txt, the first column must be voltage and the second column is current.

**Battery cycling**

Battery cycling support .xls files with state (C_CC, D_CC, R) columns from Versastudio.
Support copy-paste battery efficiency.

VE - Voltage Efficiency
CE - Coulombic Efficiency
EE - Energy Efficiency

Requires Python 3.10 and above.
Modules used: Numpy, Matplotlib, PySimpleGUI, pandas, scipy, stasmodels, and impedance.py.

License: GPLv3

Future plans.
* Rotating disc electrode
* Area under CV for gas absorption
* More file format support, if you have example files of other format, create new issue to add support.
* More flexible state recognition (C_CC, D_CC, R, etc)
* Export plot with custom dpi
* Plot EIS data, calculate parameters, and display circuit. (Note: This might be out of the scope)
* Generate Tafel plot



The file PySimpleCV/PySimpleCV contain the main code responsible to produce GUI and plotting.
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
