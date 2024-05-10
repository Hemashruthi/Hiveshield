import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def hybrid_feature_selection(self, X, y, max_iterations_da, max_iterations_abc):
    # Define the feature selection algorithms (Dragonfly Algorithm and Artificial Bee Colony)
        print("inside hybrid")

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Apply Dragonfly Algorithm (DA) for feature selection
        selected_features_da = self.dragonfly_algorithm(X_train, X_test, y_train, y_test, max_iterations_da)
        print("selected DA",selected_features_da)
        # Apply Artificial Bee Colony (ABC) for feature selection
        selected_features_abc = self.artificial_bee_colony(X_train, X_test, y_train, y_test, max_iterations_abc)

        # Combine selected features from DA and ABC (e.g., intersection or union)
        final_selected_features = np.union1d(selected_features_da, selected_features_abc)

        return final_selected_features
        
    def evaluate_features(self,features, X_train, X_test, y_train, y_test):
            print("inside evaluate")
            print(type(features))
            print(features)
            # Check the type of features
            if isinstance(features, list):
                print("features is a list")
                
                # Convert features to a list of integers
            features = features.tolist()
            print(type(features))
            print("features: ",features)
            
            # Check if all elements in the list are integers
            if all(isinstance(x, int) for x in features):
                # Check if all feature indices are within the bounds of X_train and X_test
                if all(0 <= x < X_train.shape[1] for x in features):
                    print("All feature indices are valid")
                else:
                    print("Some feature indices are out of bounds")
            else:
                print("Some elements in features are not integers")

                
            # Check if all elements in the list are integers
            '''if all(isinstance(x, int) for x in features):
                print("All elements in features are integers")
                    
                    # Check if all feature indices are within the bounds of X_train and X_test
                if all(0 <= x < X_train.shape[1] for x in features):
                    print("All feature indices are within bounds")
                else:
                    print("Some feature indices are out of bounds")
            else:
                print("Some elements in features are not integers")
            else:
                print("features is not a list")'''
            # Filter the data based on selected features
            selected_X_train = X_train[:, features]
            selected_X_test = X_test[:, features]
            print("selected_X_train: ",selected_X_train.shape)
            print("selected_X_test: ",selected_X_test.shape)
        
            # Train a Support Vector Machine (SVM) classifier
            clf = SVC(kernel='rbf')  # Use a rbf kernel for SVM
            clf.fit(selected_X_train, y_train)
        
            # Predict on test set
            y_pred = clf.predict(selected_X_test)
            print("y_pred",y_pred)
        
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print("accuracy: ",accuracy)
            return accuracy  
        
    def dragonfly_algorithm(self,X_train, X_test, y_train, y_test, max_iterations):
            print("inside dragon")
            n_features = X_train.shape[1]
    
            # Initialize dragonflies' positions (feature selection)
            positions = np.random.randint(0, 2, size=n_features) # Binary encoding for feature selection
            print("positions: ", positions)
    
            best_solution = positions
            best_accuracy = 0.0
    
            # Main iterations loop
            for iteration in range(max_iterations):
                # Evaluate current solution
                accuracy = self.evaluate_features(positions.nonzero()[0], X_train, X_test, y_train, y_test)
                print("accuracy of DA: ", accuracy)
                # Update best solution
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_solution = positions.copy()
        
                # Update positions using DA (e.g., update based on fitness value)
                # Implement DA update logic here
            print("dragon success imple")
            return best_solution  
        
    def artificial_bee_colony(self,X_train, X_test, y_train, y_test, max_iterations):
            print("inside abc")
            # Implement Artificial Bee Colony logic here
            
            n_features = X_train.shape[1]
    
            # Initialize bees' positions (feature selection)
            positions = np.random.randint(0, 2, size=n_features)  # Binary encoding for feature selection
    
            best_solution = positions
            best_accuracy = 0.0
    
            # Main iterations loop
            for iteration in range(max_iterations):
                # Evaluate current solution
                accuracy = self.evaluate_features(positions.nonzero()[0], X_train, X_test, y_train, y_test)
                print("accuracy of abc: ", accuracy)
                # Update best solution
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_solution = positions.copy()
        
            # Update positions using ABC (e.g., employed bees, onlooker bees, scout bees)
            # Implement ABC update logic here
            print("abc success")
            return best_solution   
        

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
        print((df.isnull().sum() / df.isnull().count()) * 100)
        df.dropna(inplace=True)
        print(df.isnull().sum())
        print("This Dataframe has {} rows and {} columns after removing null values".format(df.shape[0], df.shape[1]))
        
        
        print(df.columns)
        #applies a lambda function to each column (col) in the DataFrame df. This lambda function returns the unique values of each column
        print(df.apply(lambda col: col.unique()))
        numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
        print("The number of numerical features is",len(numerical_features),"and they are : \n",numerical_features)
        categorical_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
        print("The number of categorical features is",len(categorical_features),"and they are : \n",categorical_features)

        '''  # Encode IP addresses in the third column
        if df.shape[1] >= 3:
            label_encoder_3 = LabelEncoder()
            df[df.columns[2]] = label_encoder_3.fit_transform(df[df.columns[2]])

        # Encode IP addresses in the fourth column
        if df.shape[1] >= 4:
            label_encoder_4 = LabelEncoder()
            df[df.columns[3]] = label_encoder_4.fit_transform(df[df.columns[3]])
            
            
        # Assuming df is your DataFrame and protocol_column is the index of the protocol column
        protocol_column = 15  # Assuming the protocol column is at index 5

        # One-hot encode the protocol column
        df_encoded = pd.get_dummies(df, columns=[df.columns[protocol_column]], drop_first=True)

        # Drop the original protocol column
        df_encoded.drop(df.columns[protocol_column], axis=1, inplace=True)

        # Separate into features (X) and labels (y)
        X = df_encoded.drop('label', axis=1).values
        y = df_encoded['label'].values

        # Scale features using Min-Max scaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)'''

    


        # number of unique values in each numerical variable
        df[numerical_features].nunique(axis=0)
        #discrete numerical features
        discrete_feature = [feature for feature in numerical_features if df[feature].nunique()<=15 and feature != 'label']
        print("The number of discrete features is",len(discrete_feature),"and they are : \n",discrete_feature)
        df[discrete_feature].head(10)

        continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature + ['label']]
        print("The number of continuous_feature features is",len(continuous_feature),"and they are : \n",continuous_feature)
        
        for i, cat_feature in enumerate(categorical_features):
            # Iterate over each category in 'label' column
            for j, label_category in enumerate(df['label'].unique()):
                # Select data for the current label category
                data = df[df['label'] == label_category][cat_feature]
                
        # one hot encoding= Each categorical feature is transformed into multiple binary (0 or 1) columns
        print("Features which need to be encoded are : \n" ,categorical_features)
        df = pd.get_dummies(df, columns=categorical_features,drop_first=True)
        print("This Dataframe has {} rows and {} columns after encoding".format(df.shape[0], df.shape[1]))
        #dataframe after encoding
        df.head(10)
        
        X = df.drop(['label'], axis=1)
        y = df['label']

        # Scale features using Min-Max scaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        

        # Perform hybrid feature selection using DA-ABC
        selected_features = self.hybrid_feature_selection(X_scaled, y, max_iterations_da, max_iterations_abc)
        print("success of hybrid: ", selected_features)
        
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
        

        

        
    
 

    
        
    
        
        

# Example usage:
if __name__ == '__main__':
    data_path = 'D:\SEM 8\Creative and Innovative Paper\Hiveshield-main\Hiveshield-main\HIVESHIELD_Project\dataset\dataset_sdn.csv'
    
    # Initialize data preprocessor and perform preprocessing and training
    preprocessor = DataPreprocessor(data_path)
    preprocessor.preprocess_and_train()