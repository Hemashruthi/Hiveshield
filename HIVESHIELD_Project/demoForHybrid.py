import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def preprocess_and_train(self, max_iterations_da=2, max_iterations_abc=2):
        # Load data
        df = pd.read_csv(self.data_path)

        # Display basic info and handle missing values
        print(df.head(10))
        print("This Dataset has {} rows and {} columns".format(df.shape[0], df.shape[1]))
        print(df.info())
        print(df.describe())
        
        # Handle missing values
        print(df.isnull().sum())
        df.dropna(inplace=True)
        print(df.isnull().sum())
        print("This Dataframe has {} rows and {} columns after removing null values".format(df.shape[0], df.shape[1]))
        
        print(df.columns)
        print(df.apply(lambda col: col.unique()))
        
        numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
        print("The number of numerical features is", len(numerical_features), "and they are : \n", numerical_features)
        
        categorical_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
        print("The number of categorical features is", len(categorical_features), "and they are : \n", categorical_features)

        # Number of unique values in each numerical variable
        df[numerical_features].nunique(axis=0)
        
        # Discrete numerical features
        discrete_feature = [feature for feature in numerical_features if df[feature].nunique() <= 15 and feature != 'label']
        print("The number of discrete features is", len(discrete_feature), "and they are : \n", discrete_feature)
        
        continuous_feature = [feature for feature in numerical_features if feature not in discrete_feature + ['label']]
        print("The number of continuous_feature features is", len(continuous_feature), "and they are : \n", continuous_feature)
        
        # One hot encoding for categorical features
        print("Features which need to be encoded are : \n", categorical_features)
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
        print("This Dataframe has {} rows and {} columns after encoding".format(df.shape[0], df.shape[1]))
        
        X = df.drop(['label'], axis=1)
        y = df['label']

        # Scale features using Min-Max scaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform hybrid feature selection using DA-ABC
        selected_features = self.hybrid_feature_selection(X_scaled, y, max_iterations_da, max_iterations_abc)
        print("Selected features from hybrid selection:", selected_features)
        
        # Split data into train and test sets using selected features
        X_selected = X_scaled[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

        # Train an SVM classifier using the selected features
        clf = SVC(kernel='rbf', random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate the classifier
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

    def hybrid_feature_selection(self, X, y, max_iterations_da, max_iterations_abc):
        print("Inside hybrid feature selection")

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Apply Dragonfly Algorithm (DA) for feature selection
        selected_features_da = self.dragonfly_algorithm(X_train, X_test, y_train, y_test, max_iterations_da)
        print("Selected DA:", selected_features_da)

        # Apply Artificial Bee Colony (ABC) for feature selection
        selected_features_abc = self.artificial_bee_colony(X_train, X_test, y_train, y_test, max_iterations_abc)
        print("Selected ABC:", selected_features_abc)

        # Combine selected features from DA and ABC (e.g., intersection or union)
        final_selected_features = np.union1d(selected_features_da, selected_features_abc)

        return final_selected_features

    def evaluate_features(self, features, X_train, X_test, y_train, y_test):
        print("Inside evaluate features")

        if isinstance(features, np.ndarray):
            features = features.tolist()

        # Filter the data based on selected features
        selected_X_train = X_train[:, features]
        selected_X_test = X_test[:, features]

        # Train a Support Vector Machine (SVM) classifier
        clf = SVC(kernel='rbf')
        clf.fit(selected_X_train, y_train)

        # Predict on test set
        y_pred = clf.predict(selected_X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def dragonfly_algorithm(self, X_train, X_test, y_train, y_test, max_iterations):
        print("Inside Dragonfly Algorithm")
        n_features = X_train.shape[1]

        # Initialize dragonflies' positions (feature selection)
        positions = np.random.randint(0, 2, size=n_features)
        best_solution = positions
        best_accuracy = 0.0

        # Main iterations loop
        for iteration in range(max_iterations):
            accuracy = self.evaluate_features(positions.nonzero()[0], X_train, X_test, y_train, y_test)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_solution = positions.copy()

            # Update positions using DA (e.g., update based on fitness value)
            # Implement DA update logic here

        return best_solution.nonzero()[0]

    def artificial_bee_colony(self, X_train, X_test, y_train, y_test, max_iterations):
        print("Inside Artificial Bee Colony")
        n_features = X_train.shape[1]

        # Initialize bees' positions (feature selection)
        positions = np.random.randint(0, 2, size=n_features)
        best_solution = positions
        best_accuracy = 0.0

        # Main iterations loop
        for iteration in range(max_iterations):
            accuracy = self.evaluate_features(positions.nonzero()[0], X_train, X_test, y_train, y_test)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_solution = positions.copy()

            # Update positions using ABC (e.g., employed bees, onlooker bees, scout bees)
            # Implement ABC update logic here

        return best_solution.nonzero()[0]

# Example usage
if __name__ == '__main__':
    data_path = 'D:\SEM 8\Creative and Innovative Paper\Hiveshield-main\Hiveshield-main\HIVESHIELD_Project\dataset\dataset_sdn.csv'

    # Initialize data preprocessor and perform preprocessing and training
    preprocessor = DataPreprocessor(data_path)
    preprocessor.preprocess_and_train()
