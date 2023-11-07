# PySimpleCV
Graphical user interface for plotting the cyclic voltammogram and calculating jpa, jpc, and reversibility.
PySimpleCV also plots battery cycles and calculates efficiencies. Written in pure Python.

Please cite: https://doi.org/10.5281/zenodo.8019091
License: GPLv3

# Features

Feel free to make a bug report for new features

**Cyclic voltammetry**

Currently supports VersaStudio (.par), Correware(.cor), .csv, and.txt.
For.csv and .txt, the first column must be voltage and the second column is current.

Nicholson method to calculate peak current when the base line cannot be determine.
Select and plot multiple CV at the same time.
Calculate diffustion coefficient and rate of reaction from Randles-Sevcik equation.
Plot peak current vs. sqare root of scan rate for diffustion coefficient.
Plot peak current vs. peak separation for rate of reaction.
Export results and save file.

**Cyclic voltammetry ECSA**

Calculate electrochemical active surface area (ECSA) with selected area under the CV.

**Rotating Disk Electrode**

Calculate diffusion coefficient and kinetic current from Levich equation.

**Battery cycling**

Only support .xlsx from Landt Battery Test Systems. I need your help, for example, to file for more support.
Battery cycling support .xlsx files with state (C_CC, D_CC, R) columns from Landt Battery Test Systems.

VE - Voltage Efficiency

CE - Coulombic Efficiency

EE - Energy Efficiency

**Requirement**

Requires Python 3.10 and above.
Required Python modules: Numpy, Matplotlib, PySimpleGUI, pandas, statsmodels.


**Future plans**
* More file format support, if you have example files of other format, create new issue to add support.
* More flexible state recognition (C_CC, D_CC, R, etc)
* Export plot as image
* Analyze Tafel
* Slider with double handle for selecting range
* Convert between reference voltage (eg, Ag/AgCl to SHE)

The file PySimpleCV/PySimpleCV contain the main code responsible to produce GUI and plotting.
PySimpleCV/PySimpleCV_main_func.py contain mathematical function for calculation.

**Installation**

Create Python environment and install required Python modules.

`python3 -mpip install numpy matplotlib PySimpleGUI pandas statsmodels`

Clone this repository.

`git clone git@github.com:kevinsmia1939/PySimpleCV.git`

Navigate to the PySimpleCV folder and run the PySimpleCV executable. The PySimpleCV file must be marked as executable. Double click it or use the command:

`python3 PySimpleCV`

Method 2
Use Flatpak for Linux.

`flatpak install flathub io.github.kevinsmia1939.PySimpleCV`

Uninstall

`flatpak remove io.github.kevinsmia1939.PySimpleCV`

![PySimpleCV](https://github.com/kevinsmia1939/PySimpleCV/blob/main/data/screenshot/cv_screenshot.png?raw=true)
![PySimpleCV](https://github.com/kevinsmia1939/PySimpleCV/blob/main/data/screenshot/battery_screenshot.png?raw=true)
![PySimpleCV](https://github.com/kevinsmia1939/PySimpleCV/blob/main/data/screenshot/ecsa_screenshot.png?raw=true)
![PySimpleCV](https://github.com/kevinsmia1939/PySimpleCV/blob/main/data/screenshot/rde_screenshot.png?raw=true)
![PySimpleCV](https://github.com/kevinsmia1939/PySimpleCV/blob/main/data/screenshot/kou_lev_screenshot.png?raw=true)
