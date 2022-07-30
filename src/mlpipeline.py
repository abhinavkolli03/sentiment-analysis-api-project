import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

from formatter import Format
from modeling import Models
from vectorizer import VectorizeFeatures

# final pipeline that integrates the cleaning, preprocessing, training, and testing all in one
class Pipeline():
    def __init__(self, inference=False, use_count_vectorizer=False):
        #calls all the other classes in
        print("Developing pipeline...")
        self.inference = inference
        self.format = Format()
        self.vector_features = VectorizeFeatures()
        self.models = Models()
        self.models.create_estimators()
        #if this is just using the API to predict, then load pre-existing vectorizer
        if self.inference:
            self.vectorizer = joblib.load(open("../vectorizer.pkl", "rb"))
        #otherwise, create and train a new one
        #you can also set the use_count_vectorizer to call the CountVectorizer instead,
        #but the TFIDF Vectorizer is default for now
        else:
            if use_count_vectorizer:
                self.vectorizer = CountVectorizer(stop_words='english')
            else:
                self.vectorizer = TfidfVectorizer(stop_words='english')

    def manipulate_content(self, data, inference=False):
        #reformats the data into content that can be easily vectorized and trained on
        data = self.format.adjust(data)
        #retrieves the new vector features from formatted text
        vector_features = self.vector_features.vectorize(data["formatted_text"], self.vectorizer, inference)
        if inference:
            #if this is just a test, then only return the vector features
            return vector_features
        #we need airline sentiment data as well for the y_train and y_test of training/testing phase
        return vector_features, data["airline_sentiment"]

    def split_data(self, X, y, test_size):
        #split content appropriately (default test size of .15)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
        return X_train, X_test, y_train, y_test

    def grid_search_models(self, content, test_size=0.15):
        #main function of the pipeline that includes: cleaning, preprocessing, train, test
        X, y = self.manipulate_content(content)
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size)
        trained_estimators = self.models.train_grid(X_train, y_train)
        self.models.test_grid(trained_estimators, X_test, y_test)

    def create_inference(self, model, data):
        #in case this is a predict API call, we use this function to test already trained models
        vector_features = self.manipulate_content(data, inference=True)
        predictions = self.models.infer(model, vector_features)
        return predictions