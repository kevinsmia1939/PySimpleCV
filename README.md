# PySimpleCV

PySimpleCV are being rewritten with PyQt5 called PySimpleEChem https://github.com/kevinsmia1939/PySimpleEChem
This should bring support for Wayland on Linux, more modern looking interface, and move away from now propiertary PySimpleGUI.

Graphical user interface for plotting and analyzing the cyclic voltammograms (CV) with simple to use sliders. <br />
PySimpleCV also plots battery cycles and calculates efficiencies. Written in pure Python. <br />

Please cite: https://doi.org/10.5281/zenodo.8019091 <br />
License: GPLv3 <br />

# Features
Feel free to make a bug report for new features

**Cyclic voltammetry**

* Select and plot multiple CV at the same time.<br />
* Currently supports VersaStudio (.par), Correware(.cor), .csv, and .txt. Please send some other format for more file format support. For .csv and .txt, the first column must be voltage and the second column is current.<br />

* Nicholson method to calculate peak current when the base line cannot be determine.<br />
* Calculate diffustion coefficient and rate of reaction from Randles-Sevcik equation.<br />
* Plot peak current vs. sqare root of scan rate for diffustion coefficient.<br />
* Plot peak current vs. peak separation for rate of reaction.<br />
* Detect peak with maximum, minimum, or 2nd derivatives.<br />
* Export results and save file.<br />

**Cyclic voltammetry ECSA**

* Calculate electrochemical active surface area (ECSA) with selected area under the CV.

**Rotating Disk Electrode**

* Calculate diffusion coefficient and kinetic current from Levich equation.

**Battery cycling**

* Only support .xlsx from Landt Battery Test Systems. I need your help, for example, to file for more support.<br />
* Battery cycling support .xlsx files with state (C_CC, D_CC, R) columns from Landt Battery Test Systems.<br />

VE - Voltage Efficiency<br />
CE - Coulombic Efficiency<br />
EE - Energy Efficiency<br />

**Requirement**

* Requires Python 3.10 and above.<br />
* Required Python modules: Numpy, Matplotlib, PySimpleGUI, pandas, statsmodels.<br />


**Future plans**
* More file format support, if you have example files of other format, create new issue to add support.
* More flexible state recognition (C_CC, D_CC, R, etc)
* Export plot as image
* Analyze Tafel
* Slider with double handle for selecting range
* Convert between reference voltage (eg, Ag/AgCl to SHE)

The file PySimpleCV/PySimpleCV contain the main code responsible to produce GUI and plotting.<br />
PySimpleCV/PySimpleCV_main_func.py contain mathematical functions for calculation.<br />

# Installation
Method 1

1\. Create Python environment.<br />
2\. Install required Python modules.<br />

`python3 -mpip install numpy matplotlib PySimpleGUI pandas statsmodels`

3\. Clone this repository.<br />

`git clone git@github.com:kevinsmia1939/PySimpleCV.git`

4\. Navigate to the PySimpleCV folder and run the PySimpleCV executable. The PySimpleCV file must be marked as executable. Double click it or use the command:<br />

`python3 PySimpleCV`

Method 2 <br />

Use Flatpak for Linux.<br />

`flatpak install flathub io.github.kevinsmia1939.PySimpleCV`

Uninstall <br />

`flatpak remove io.github.kevinsmia1939.PySimpleCV`

![PySimpleCV](https://github.com/kevinsmia1939/PySimpleCV/blob/main/data/screenshot/cv_screenshot.png?raw=true)
![PySimpleCV](https://github.com/kevinsmia1939/PySimpleCV/blob/main/data/screenshot/battery_screenshot.png?raw=true)
![PySimpleCV](https://github.com/kevinsmia1939/PySimpleCV/blob/main/data/screenshot/ecsa_screenshot.png?raw=true)
![PySimpleCV](https://github.com/kevinsmia1939/PySimpleCV/blob/main/data/screenshot/rde_screenshot.png?raw=true)
![PySimpleCV](https://github.com/kevinsmia1939/PySimpleCV/blob/main/data/screenshot/kou_lev_screenshot.png?raw=true)
