import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

# Read the dataset
df = pd.read_csv('D:/SEM 8/Creative and Innovative Paper/Hiveshield-main/Hiveshield-main/HIVESHIELD_Project/dataset/dataset_sdn.csv')

# Display the first 10 rows
print(df.head(10))

# Display the dimensions of the dataframe
print("This Dataset has {} rows and {} columns".format(df.shape[0], df.shape[1]))

# Display information about the dataframe
print(df.info())

# Display summary statistics of the dataframe
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Calculate the percentage of missing values for each column
print((df.isnull().sum()/df.isnull().count())*100)

# Remove rows with missing values
df.dropna(inplace=True)

# Display the number of missing values after removal
print(df.isnull().sum(), 'missing values')

# Display the final dimensions of the dataframe
print("This Dataframe has {} rows and {} columns after removing null values".format(df.shape[0], df.shape[1]))

# Separate features and labels
X = df.drop('label', axis=1)  # Features
y = df['label']  # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dragonfly Algorithm parameters
MaxGen = 100
FN = 10
Prob = 0.1

# Initialize dragonflies' population and step vectors
x = np.random.rand(FN, X.shape[1])  # Dragonflies' population
Axi = np.zeros_like(x)  # Step vectors

# Artificial Bee Colony parameters
limit = 5
Sep_weight = 1.5
Align_weight = 1.0
Coh_weight = 1.0
Food_attr_weight = 1.0
Enemy_dist_weight = 1.0

# Define the Dragonfly Bee Phase
def Dragonfly_Bee_Phase():
    # Perform feature selection using the hybrid of ABC and DA
    selected_features = hybrid_ABC_DA()

    # Train an SVM classifier using the selected features
    svm_classifier = svm.SVC()
    svm_classifier.fit(X_train_scaled[:, selected_features], y_train)

    # Use the classifier to classify the data
    predicted_labels = svm_classifier.predict(X_test_scaled[:, selected_features])

    # Calculate the accuracy of the classifier
    accuracy = np.mean(predicted_labels == y_test)

    return accuracy

# Define the Onlooker Bee Phase
def Onlooker_Bee_Phase():
    # Perform feature selection using the hybrid of ABC and DA for selected dragonflies
    for i in range(FN):
        selected_features = hybrid_ABC_DA()
        
        # Train an SVM classifier using the selected features
        svm_classifier = svm.SVC()
        svm_classifier.fit(X_train_scaled[:, selected_features], y_train)
    
        # Use the classifier to classify the data
        predicted_labels = svm_classifier.predict(X_test_scaled[:, selected_features])
    
        # Calculate the accuracy of the classifier
        accuracy = np.mean(predicted_labels == y_test)

# Define the hybrid of ABC and DA for feature selection
def hybrid_ABC_DA():
    # Implement your hybrid algorithm here
    # This is a placeholder function and needs to be replaced with your implementation
    selected_features = np.random.choice(X.shape[1], 5, replace=False)  # Select 5 random features
    return selected_features

# Main loop of the algorithm
iter = 0
while iter < MaxGen:
    for i in range(FN):
        if np.random.rand() < Prob:
            Dragonfly_Bee_Phase()
        else:
            Onlooker_Bee_Phase()
    
    iter += 1

print(f'Final Accuracy: {Dragonfly_Bee_Phase()}')
