import joblib
# class to vectorize features
class VectorizeFeatures:
    def __init__(self):
        print("Vectorizing features...")

    def vectorize(self, data, vectorizer, inference):
        # checks if this is an inference test
        if inference:
            #uses pre-trained model
            vectors = vectorizer.transform(data).toarray()
        else:
            # fits to the list of words and then transforms
            vectors = vectorizer.fit_transform(data).toarray()
            # writes into the system the new vectorizer
            joblib.dump(vectorizer, "../vectorizer.pkl")
        return vectors