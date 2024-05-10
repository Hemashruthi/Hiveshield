import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv('D:\SEM 8\Creative and Innovative Paper\Hiveshield-main\Hiveshield-main\HIVESHIELD_Project\dataset\dataset_sdn.csv')
X = df.drop('label', axis=1).values
y = df['label'].values
#print(df.head())
#print(X, 'X')
#print(y, 'y')

def abc_feature_selection(X, y, num_iterations, num_selected_features):
    num_features = X.shape[1]
    best_solution = 1.0
    best_fitness = float('-inf')

    for _ in range(num_iterations):
        # Generate a random solution (subset of selected features)
        solution = np.random.choice([0, 1], size=num_features)
        selected_indices = np.nonzero(solution)[0]
        #print(solution)
        #print(selected_indices)

        # Evaluate fitness of the solution using SVM and accuracy
        X_selected = X[:, selected_indices]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, svm_model.predict(X_test))

        # Update best solution if current solution is better
        if accuracy > best_fitness:
            best_solution = solution.copy()
            best_fitness = accuracy

    # Select top features based on best solution
    selected_indices = np.nonzero(best_solution)[0][:num_selected_features]
    return selected_indices
def da_feature_selection(X, y, num_iterations, num_selected_features):
    num_features = X.shape[1]
    positions = np.random.rand(num_iterations, num_features)
    velocities = np.zeros((num_iterations, num_features))

    best_solution = 1.0
    best_fitness = float('-inf')

    for i in range(num_iterations):
        # Update velocities
        r1 = np.random.rand(num_features)
        r2 = np.random.rand(num_features)
        velocities[i] += r1 * (best_solution - positions[i]) + r2 * (best_solution - positions[i])

        # Update positions
        positions[i] += velocities[i]

        # Evaluate fitness of the solution using SVM and accuracy
        solution = (positions[i] > 0.5).astype(int)
        selected_indices = np.nonzero(solution)[0]

        X_selected = X[:, selected_indices]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, svm_model.predict(X_test))

        # Update best solution if current solution is better
        if accuracy > best_fitness:
            best_solution = solution.copy()
            best_fitness = accuracy

    # Select top features based on best solution
    selected_indices = np.nonzero(best_solution)[0][:num_selected_features]
    return selected_indices
# Example: Load your dataset (replace with your data)
X = np.random.rand(100, 20)  # Example data (100 samples, 20 features)
y = np.random.randint(0, 2, size=100)  # Random labels (binary classification)

# Apply ABC or DA for feature selection
selected_indices_abc = abc_feature_selection(X, y, num_iterations=100, num_selected_features=5)
selected_indices_da = da_feature_selection(X, y, num_iterations=100, num_selected_features=5)

print("Selected features using ABC:", selected_indices_abc)
print("Selected features using DA:", selected_indices_da)