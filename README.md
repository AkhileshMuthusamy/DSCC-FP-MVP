# DSCC-FP-MVP

This project predicts the stock market prices when a user provides historical data of an organization which helps the user to buy or sell the stocks using AI and ML-based models. 

Stock market prediction has been advanced since the beginning of Machine Learning that has started a new era in the field of technology. There are many approaches to predict stock market data based on the size of data and its features. 

Our objective is to build a solution to scrape the stock data from yahoo finance to help the investors by giving predictions. Stock price and trends prediction are challenging as the market is dynamic. Applying machine learning to the data collected, we are trying to predict the price and trends. With the help of the proposed solution, we can help the investors by offering better financial pieces of advice.

## Technology Stack:
 
The MVP project has four main components: frontend, backend, ML model and database.

* Frontend - Web App: Angular / React.js
* Backend - Python: Flask
* ML Model - Python: Jupyter notebook
* Database - MongoDB / Firebase

# Environment

* Development - `dev` branch
* Testing - `uat` branch
* Production - `main` branch

# Setup Virtual Environment for Python

1. Clone the repository.
2. Using command prompt navigate to the root folder of the project.
3. Create the virtual environment `python -m venv .venv`
4. Activate the venv. Open the file in command prompt `.venv\Scripts\activate.bat`
5. Install dependencies from requirements.txt file `pip install -r requirements.txt`
6. If you include any new packages to the projects, then update the requirements.txt file using `pip freeze > requirements.txt`
