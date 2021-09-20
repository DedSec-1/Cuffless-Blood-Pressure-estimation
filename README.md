# Heart-Disease-Prediction

<img align="right" width="210" height="210" src="https://image.slidesharecdn.com/cufflessbloodpressuremonitoringfinalppt-150614095256-lva1-app6892/85/cuffless-blood-pressure-monitoring-project-1-320.jpg?cb=1434275625">

<p>Conventional blood pressure (BP) measurement methods have different drawbacks such as being invasive, cuff-based, or requiring manual operations. 
Sometimes we require continuous measurement of Blood Pressure, which is not possible with our traditional <strong><i>SPHYGMOMANOMETER</i></strong>.</p>
In this project, we are trying to implement a non-invasive method of continuous BP measurement using machine learning models, and later on, we will use extracted features from ECG and PPG signals for the prognosis of various heart-related diseases.<br>
For the reference, we are following these research papers:

1. Non‑invasive cuff‑less blood pressure estimation using
a hybrid deep learning model(<a href="https://link.springer.com/article/10.1007/s11082-020-02667-0">Link</a>).


## Steps to run code
1. `pip install -r requirements.txt` : Installs the required python libraries
2. `python ./preprocessing/feature_extraction.py` : Run the Feature extraction file and exports *features_20.csv* in output folder.
3. `python ./preprocessing/PTT_final.py` : Process the data points and find PTT (Pulse Transit Time) and exports *ptt_feature.csv* in output folder.
4. You will have the required csvs.