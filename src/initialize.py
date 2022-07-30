#class for creating, training, and testing pre-trained models
import pandas as pd
import joblib
from mlpipeline import Pipeline

if __name__ == "__main__":
    #uses airline data to initialize, train, and test models
    data = pd.read_csv("../data/airline_sentiment_analysis.csv")
    sentiment = Pipeline()
    sentiment.grid_search_models(data)