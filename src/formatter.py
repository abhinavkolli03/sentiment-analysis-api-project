import re
import string

# class to preprocess data
class Format:
    def __init__(self):
        print("Formatting inputs...")

    def find_tag(self, text):
        # finds and returns the text after the @ (the airline)
        mention = re.findall(r"@([A-Za-z]+)", text)[0]
        return mention

    def delete_tag(self, text):
        # removes airline tag in formatted_text
        mention = re.findall(r"@([A-Za-z]+)", text)[0]
        text = str(text).replace(mention, '')
        return text

    def format_text(self, text):
        # joins chars in string punctuation list
        text = "".join([c for c in text if c not in string.punctuation])
        # removes links
        text = re.sub(r'http\S+', '', text)
        # removes all special characters but spaces
        text = re.sub(r'\W+', ' ', str(text))
        # removes numbers
        text = re.sub("[0-9]+", "", text)
        # removes single letter chars
        text = re.sub(r'\b[a-zA-Z]\b', '', text)
        # replaces big spaces with one space
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        text = re.sub(r'\s+$', '', text)
        text = text.lower()
        return text

    def adjust(self, data):
        # a new column to describe the airline in the review
        data["airline"] = data["text"].apply(lambda txt: self.find_tag(txt))
        # removes the airline name and tag since it's in a new column
        data["formatted_text"] = data["text"].apply(lambda txt: self.delete_tag(txt))
        # reformats all the text to fit the parameters of token vectorizer
        data["formatted_text"] = data["formatted_text"].apply(lambda txt: self.format_text(txt))
        return data