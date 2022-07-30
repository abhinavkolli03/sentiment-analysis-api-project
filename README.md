# sentiment-analysis-api
Sentiment analysis API that identifies whether an airline review has positive or negative sentiment.

I designed this project to test how several binary classification models would perform against an airline reviews dataset. The goal of each model was to see whether the models could accurately predict the review's sentiment: positive or negative. Additionally, this microservice structure of each class helped automate training and testing many models.

Project Demo through browser API calls...
https://vimeo.com/734881463

# Classification Models Tested:
- Logistic Regression
- K-Nearest Neighbors
- LinearSVC
- Multinomial Naive Bayes
- Random Forest
- Decision Tree
- Gradient Boosting

# Py File Descriptions
- [formatter.py](url): Takes in dataset inputs and "cleans" the current data. First, it identifies where the tag in the review tweet is and stores that into a new column called "airline," which can be used to organize the reviews to the airline. Then, the class deletes the tag and restructures the review into a new column called "formatted_text" after cleaning out unnecessary characters, spaces, and single-letter words.
- [vectorizer.py](url): Takes in formatted data and generates vectorized features using TfIdf or CountVectorizer. Saves vectorizer into vectorizer.pkl.
- [modeling.py](url): Stores list of parameter grids and their respective models. It also contains a train_grid and test_grid funciton that contains the main functions to train/test the models. This is the main area for hyperparameter tuning and data analysis of model results.
- [mlpipeline.py](url): Main pipeline that calls and uses all the other classes from the file to automate ML training/testing. It cleans the data, then preprocesses, and lastly trains and tests the models using the data.
- [fastapiserver.py](url): Quick implementation of FastAPI that enabled the following APIs:
  - initialize: initializes and recreates all binary classifier models from scratch.
  - predict: by providing a review and available model, the API can process a prediction and return the sentiment back.
- [inference.py](url) and [initialize.py](url): tester classes used during creation of API
- [pyramidswaggerserver.py](url): Implemented swagger.yaml schema to create a public endpoint for the API through Pyramid Swagger UI. The swagger implementation works on the backend in terms of calling APIs through browser, but the Swagger UI portion is still under maintenance.

# Model Train/Test Results
Logistic Regression
![image](https://user-images.githubusercontent.com/85178092/181861951-b80b9651-5c38-4c80-8b7f-0dce1ed24a94.png)

K-Nearest Neighbors
![image](https://user-images.githubusercontent.com/85178092/181862016-7d1edc46-94b4-47f7-ae9d-03318f2cfc4b.png)

Linear Support Vector Classifier
![image](https://user-images.githubusercontent.com/85178092/181862065-2e5e5c06-108e-422d-be58-3c59224dea03.png)

Multinomial Naive Bayes
![image](https://user-images.githubusercontent.com/85178092/181862111-988b8387-337c-4391-b115-f391f5035d0e.png)

Decision Tree Classifier
![image](https://user-images.githubusercontent.com/85178092/181862087-ea7fd536-e1a4-4288-a794-ebae0a70e18f.png)

# Hyperparameter Tuning
I implemented grid search for each of my models to try out different hyperparameters automatically in my mlpipeline.
