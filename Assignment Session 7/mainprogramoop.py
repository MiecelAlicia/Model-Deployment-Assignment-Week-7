import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.x = None
        self.y = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)

    def preprocess_data(self):
        self.df.drop(columns=['Id'], inplace=True, errors='ignore')  # Drop 'Id' if exists
        self.df = self.df.drop_duplicates()

        label_encoder = LabelEncoder()
        self.df['Species'] = label_encoder.fit_transform(self.df['Species'])

        self.x = self.df.drop(columns=['Species'])
        self.y = self.df['Species']

        return self.x, self.y

class ModelHandler:
    def __init__(self, x, y, test_size=0.2, random_state=42):
        self.x = x
        self.y = y
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=random_state
        )
        self.model = RandomForestClassifier(random_state=random_state)  # Default Random Forest Model

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions)

    def make_prediction(self):
        self.y_predict = self.model.predict(self.x_test)

    def create_report(self):
        print('\nClassification Report - Random Forest\n')
        print(classification_report(self.y_test, self.y_predict, target_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']))

    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

# Example Usage
file_path = 'Iris.csv'
data_handler = DataHandler(file_path)
data_handler.load_data()
x, y = data_handler.preprocess_data()

model_handler = ModelHandler(x, y)
model_handler.train_model()

print("Model Accuracy:", model_handler.evaluate_model())
model_handler.make_prediction()
model_handler.create_report()

# Save the trained model
model_handler.save_model_to_file('random_forest_model.pkl')
