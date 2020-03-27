# Disaster_Response_Pipeline

This project aims to classify messsages at the time of a disaster into one of 36 categories. The dataset is provided by [Figure Eight](https://www.figure-eight.com). A Random Forest Classifier is used to classify the message into one of the categories.

### The Project consists of the following steps
1. Data Preprocessing: Involves analyzing and cleaning the data (removing duplicates, Removing NaN values, separating columns into features). Further details and code are available in the [ETL Notebook](ETL Pipeline Preparation.ipynb)
2. Model Training: Features were generated and a Classifier was trained on the data using a pipeline. For further details and code, see [ML Notebook](ML Pipeline Preparation.ipynb)
3. Web Application: Deploy a web app to use the trained model to classify messages.

### Instructions
1. Run the following commands in the project's root directory to set up the database and the model.

    - To run the ETL pipeline that cleans data and stores in database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run the ML pipeline that trains classifier and save it:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ to see the web app.

### Files
- app/ -> Folder containing files used by Web Application
<br>-- run.py -> Python script to run the web app
<br>-- templates/ -> Folder containing HTML for the Web Page
<br>--- go.html -> HTML for webpage
<br>--- master.html -> HTML for webpage
<br>-- clf.pkl -> Stored Machine Learning model
- data/ -> Folder containing the dataset and Python scripts for Data Preprocessing
<br>-- disaster_categories.csv -> Dataset
<br>-- disaster_messages.csv -> Dataset
<br>-- process_data.py -> Python script to clean and store data
<br>-- DisasterResponse.db -> Stored SQL Database
- models/ -> Folder Containing Python Script to create and save Classifier model
<br> -- train_classifier.py -> Python Script to create and save Classifier model
- ETL Pipeline Preparation.ipynb -> Notebook containing code for data preparation
- ML Pipeline Preparation.ipynb -> Notebook containing code for model creation
