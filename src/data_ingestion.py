import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataIngestionLayer:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.data = self.load_dataset()
        self.preprocessed_data = self.preprocess_data()

    def load_dataset(self):
        if self.dataset_name == "PaySim":
            data = pd.read_csv("paysim.csv")  # Replace with actual path
        elif self.dataset_name == "CreditCard":
            data = pd.read_csv("creditcard.csv")  # Replace with actual path
        elif self.dataset_name == "IoT-23":
            data = pd.read_csv("iot23.csv")  # Replace with actual path
        else:
            raise ValueError("Invalid dataset name")
        return data

    def preprocess_data(self):
        scaler = StandardScaler()
        numerical_features = self.data.select_dtypes(include=[np.number]).columns
        self.data[numerical_features] = scaler.fit_transform(self.data[numerical_features])
        return self.data