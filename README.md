# RPyCA
This repository is currently being developed.

As files for the data sets being used are too large to host on Github, they can be downloaded at the links below.
### Main data set: 
https://www.unb.ca/cic/datasets/ids-2017.html

This now redirects to a form. If you want to skip the form you can go here:
http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/
This link was valid as of July 17, 2022.

Download GeneratedLabelledFlows.zip, decompress it, and move Traffic Labelling to datasets folder. "TrafficLabelling " has a space at the end that must be deleted.

There is a UTF-8 error msg using that data.
Using MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv file as replacement has KeyError

Opening "TrafficLabelling" form of file in libre then resaving as CSV fixed UTF error.

### Secondary data set (LLDOS 2.0.2 - Scenario Two):
https://www.ll.mit.edu/r-d/datasets/2000-darpa-intrusion-detection-scenario-specific-datasets

This started as a research project of investigating whether Robust PCA can be used to improve the performance of neural networks. Currently, it is being developed to become a Python package for further application. Once completed, users will be able to import the package into a python file, and run the commands to enhance their dataset(s). Output will include training, testing, and validation sets along with their corresponding sets for labels.

# Dependencies
see requirements.txt, python3

# Usage
Before first use run this:
````
mkdir logs
````


Then run this:
````
python3 main.py MAIN
````
The argument after main.py is which config in config.ini to use.

There is console output of results while it runs and logs are put in logs/resultsSVM.log. Note this file is overwritten if you rerun the program.
