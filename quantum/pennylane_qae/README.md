## PennyLane+PyTorch QAE Example

- **Author:** Jacob Cybulski
- **Date:** June 2024
- **Aims:** *To illustrate PL+Torch development of hybrid quantum/classical models*

### Folders:
- beer_pl: which is a log (or a database) of data, models and analyses
- utils: which is a collection of Python files and testing notebooks to manage time series, plots and quantum circuits.</br>
   Note that some of these files were produces for Qiskit and they need to be cleaned up.
  
### Important notebooks:
- ts_pl_v4_03_data_beer.ipynb: allowing to create a data set in a log directory
- ts_pl_tlay_v4_03_train_lat7.ipynb: trains and saves a sample PL+PT hybrid QAE model
- ts_pl_tlay_v4_03_analysis_lat7.ipynb: analyses the QAE model retrieved from logs

### Other notebooks:
- ts_pl_versions.ipynb: description of different versions (useless and most relate to Qiskit software)
- ts_torch_v4_01_training_lat7.ipynb: my previous classical PyTorch QAE from which a PL+PT solution was adopted
- ts_torch_v4_01_analysis_lat7_bw.ipynb: the analysis of the my previous classical Torch QAE

### Important utilities:
- Circuits.py / Test_TS_Circuits.ipynb: shows creation of various QAE parts in PennyLane
- Test_TS_PennyLane.ipynb: a series of examples how to use PennyLane+Torch