# CSE6242 EIA 930 Forecast Analysis App User Guide
Welcome to the EIA 930 Forecast Analysis App
Included in this repo is all the code needed to run the EIA 930 Forecast Analysis App locally. You can also visit https://cse6242-eia-demand-forcast.streamlit.app/ to use the app anywhere. 

# Description
Below is a breakdown of the files and folders included:

**pages folder:** Includes all of the files needed to populate the app pages. The name of the file is the name of the page in the app.

**data folder:** Includes all of the data used for evaluations, predicitions and visualizations. Larger datasets are stored as zip files and are unzipped automatically in the app.

**models folder:** Includes all of the machine learning models used in project. All files are saved as **pkl** files and are compressed with **gzip** to save space. All models are unzipped automatically in the app.

**app.py:** The main python file used to generate the app. Sets the home screen and page selection

**EIA930_Analysis.ipynb:** Jupyter notebook where all of the data processing and machine learning training/evaluation is done

**requirements.txt:** Packages required to run app

# Installation: 
1. Clone the repository
2. Create enviroment using conda: `conda create --name CSE6242-Project-Env python=3.9.2`
3. `conda activate CSE6242-Project-Env`
4. Install the required packages using the following command: `pip install -r requirements.txt`
5. Run streamlit app using the following command: `streamlit run app.py`

# Execution
Navigate to desired page to view content. If you want to predict the load for LAWP go to predictions page and fill in required data. 

# Demo
https://youtu.be/A2_w1u00YAg