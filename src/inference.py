#class for testing inferences into pre-trained models
import pandas as pd
import joblib
from mlpipeline import Pipeline

if __name__ == "__main__":
    #load dataframe consisting of test values here
    data = pd.read_csv("../data/airline_sentiment_analysis.csv")
    tested_model = joblib.load("../models/LogisticRegression.pkl")
    sentiment = Pipeline(inference=True)

    #load individual test values inside here
    test_data = pd.DataFrame(['@VirginAirlines I love this flight!'], columns=['text'])
    predictions = sentiment.create_inference(tested_model, test_data)
    print(predictions)