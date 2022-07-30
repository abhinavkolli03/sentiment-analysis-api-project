import pandas as pd
from fastapi import FastAPI
import uvicorn
import joblib
from mlpipeline import Pipeline

app = FastAPI(debug=True)

#tests to see if fastapi works
@app.get('/')
def home():
    return {'text': 'Server works!'}

#api for predictions
@app.post('/predict')
def predict_text(Sentiment: str, Selected_Model: str):
    #loads preferred model
    tested_model = joblib.load(f"../models/{Selected_Model}.pkl")
    #sets text into df to be fed into model
    sentiments_df = pd.DataFrame(pd.Series(Sentiment), columns=["text"])
    #creates pipeline sentiment and makes an inference
    sentiment = Pipeline(inference=True)
    prediction = sentiment.create_inference(tested_model, sentiments_df)
    #if positive, then JSON about positive sentiment will be returned; otherwise, negative
    if prediction == "positive":
        return {"Review contains positive sentiment"}
    return {"Review contains negative sentiment"}

#API to retrain and initialize models
@app.get('/initialize')
def initialize():
    #uses airline data
    data = pd.read_csv("../data/airline_sentiment_analysis.csv")
    #creates pipeline and grid search trains and tests all binary classifiers
    sentiment = Pipeline()
    sentiment.grid_search_models(data)
    return {"Completed training/testing all models and vectorizers"}

if __name__ == "__main__":
    #backend server
    uvicorn.run(app)