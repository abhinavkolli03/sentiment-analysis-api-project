#importing required packages
from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response
from pyramid.view import view_config

from mlpipeline import Pipeline
import pandas as pd
import joblib

#test to check if server works
@view_config(route_name="test")
def test(request):
    return Response("Server works")

@view_config(route_name="initialize")
def initialize(request):
    #API call to initialize and train/test all models
    data = pd.read_csv("../data/airline_sentiment_analysis.csv")
    sentiment = Pipeline()
    sentiment.grid_search_models(data)
    return Response("Completed training/testing all models and vectorizers")

@view_config(route_name="predict", request_method="POST")
def predict(request):
    #gathers the text from API call and feeds into the pipe for prediction
    text = request.swagger_data['review_text']
    sentiments_df = pd.DataFrame(pd.Series(text), columns=["text"])
    prediction = pipe.create_inference(model, sentiments_df)
    #returns sentiment
    if prediction == "positive":
        return Response("Review contains positive sentiment")
    return Response("Review contains negative sentiment")


if __name__ == "__main__":
    settings = {}
    print("Swagger UI pyramid still poses an error, but the pyramid chameleon works with Postman and HTTP requests")
    #fix the pyramid-swagger configuration

    #settings to edit the properties of the configurator
    settings['pyramid_swagger.enable_swagger_spec_validation'] = True
    settings['pyramid_swagger.schema_directory'] = 'api_docs'
    settings['pyramid_swagger.schema_file'] = 'swagger.yaml'
    settings['pyramid_swagger.include_missing_properties'] = True

    #choose random model to load here; pipeline will be created for it
    model = joblib.load(open("../models/LogisticRegression.pkl", "rb"))
    pipe = Pipeline(inference=True)

    #passes pyramid requirements and routes to config
    with Configurator() as config:
        config = Configurator(settings=settings)
        config.include('pyramid_chameleon')
        #config.include('pyramid_swagger')
        config.add_route("predict", "/predict")
        config.add_route("test", "/")
        config.add_route("initialize", "/initialize")
        config.scan()
        #creates app server
        app = config.make_wsgi_app()
    server = make_server("0.0.0.0", 8080, app)
    server.serve_forever()