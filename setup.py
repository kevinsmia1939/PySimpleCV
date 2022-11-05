from setuptools import find_packages
from setuptools import setup
setup(
   name='PySimpleCV',
   version='1.3',
   description='Small GUI software to plot CV and battery cycles',
   author='Kavin Teenakul',
   author_email='kevin_tee@protonmail.com',
   url="https://github.com/kevinsmia1939/PySimpleCV",
   packages=['PySimpleCV'],  #same as name
   install_requires=['numpy', 'pandas', 'PySimpleGUI', 'scipy', 'matplotlib'],
   scripts=['PySimpleCV/PySimpleCV.py', 'PySimpleCV/PySimpleBattery.py']
)
