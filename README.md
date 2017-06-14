# Event Detection in Twitter
###About
This repository contains code used for the Semester Project: "Benchmarking specific and general event detection aproaches" for the LSIR Lab at EPFL. 

###Dependancies
Please install the following:
```
pip3 install numpy
pip3 install scipy
pip3 install tqdm
pip3 install PyWavelets
pip3 install python-igraph
pip3 install -U scikit-learn
pip3 install humanize
pip3 install pandas
pip3 install pympler
pip install sparselsh
```

###Data
The datasets can be found in the /data directory. Note: randomTweets.csv was too large to commit on GitHub, and so was the compressed zip file. So please concatinate the 4 files randomTweets1.csv, randomTweets2.csv, randomTweets3.csv, and randomTweets4.csv into one file called randomTweets.csv before using the code provided.

###Running the code
In order to test the code, you can do the following:
#### FeatureTracking:
```
python3 runFeatureTracking.py <optional file name>
```
By default featureTracking will run on the ManchesterAttack dataset, this can be changed by passing a different file name as argument. Note, please change the following variables based on the file you are choosing:

`FLAG`: To manually set the boundary between important and unimportant events, set FLAG to your selected value. If the FLAG is set to 0 or a negative number featureTrajectories will set it based on the heuristics of the stopwords. For Manchester

`bucketSize`: This refers to the time duration that each bucket spans. Please change the values based on the dataset being used. (1=seconds,60=minutes,3600=hours,86400=days). For ManchesterAttacks set bucketSize to 60, and to all other datasets provided in this repository set it to 86400 for meaningful results.

In order to test the Real Attacks dataset, please uncomment lines 28-41, and comment out lines 15-25 and 56

#### SigniTrend:
```
python3 runSigniTrend.py <optional file name>
```
By default SigniTrend will run on the ManchesterAttack dataset, this can be changed by passing a different file name as argument. Note, please change the following variables based on the file you are choosing:

`bucketSize`: This refers to the time duration that each bucket spans. Please change the values based on the dataset being used. (1=seconds,60=minutes,3600=hours,86400=days). For ManchesterAttacks set bucketSize to 60, and to all other datasets provided in this repository set it to 86400 for meaningful results.

In order to test the Real Attacks dataset, please uncomment lines 34-53, and comment out lines 14-31.

In order to plot the results from SigniTrend, there are several commented out sections of code that can be used.


#### EDCoW:
```
python3 runEDCoW.py <optional file name>
```
By default EDCoW will run on the ManchesterAttack dataset, this can be changed by passing a different file name as argument. Note, please change the following variables based on the file you are choosing:

`bucketSize`: This refers to the time duration that each bucket spans. Please change the values based on the dataset being used. (1=seconds,60=minutes,3600=hours,86400=days). For ManchesterAttacks set bucketSize to 60, and to all other datasets provided in this repository set it to 86400 for meaningful results.

#### RealTime:
```
python runRealTime.py <optional file name>
```
Note: In order to test realTime, you will have to use Python2, and not Python3 as I had installation issues with the Python3 version of Sparselsh library.

By default RealTime will run on the ManchesterAttack dataset, this can be changed by passing a different file name as argument. Note, please change the following variables based on the file you are choosing:

`bucketSize`: This refers to the time duration that each bucket spans. Please change the values based on the dataset being used. (1=seconds,60=minutes,3600=hours,86400=days). For ManchesterAttacks set bucketSize to 60, and to all other datasets provided in this repository set it to 86400 for meaningful results.