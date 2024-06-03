import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif

class DragonflyAlgorithm:
    def __init__(self, n_features, n_dragonflies=10, max_iterations=10, w=0.9, c=0.1, s=0.1):
        self.n_features = n_features
        self.n_dragonflies = n_dragonflies
        self.max_iterations = max_iterations
        self.w = w
        self.c = c
        self.s = s
        print("Default values initialised for dragonfly algorithm")

    def evaluate_features(self, features, X_train, X_test, y_train, y_test):
        if len(features) == 0:
            return 0
        selected_X_train = X_train[:, features]
        selected_X_test = X_test[:, features]
        # Compute mutual information scores for selected features
        mi_scores = mutual_info_classif(selected_X_train, y_train)
        # Return the mean of mutual information scores as fitness value
        return np.mean(mi_scores)

    def optimize(self, X_train, X_test, y_train, y_test):
        positions = np.random.randint(0, 2, size=(self.n_dragonflies, self.n_features)).astype(float)
        velocities = np.random.uniform(-1, 1, size=(self.n_dragonflies, self.n_features))
        best_solution = positions[0]
        best_fitness = 0.0

        for iteration in range(self.max_iterations):
            for i in range(self.n_dragonflies):
                features = np.where(positions[i] >= 0.5)[0]
                fitness = self.evaluate_features(features, X_train, X_test, y_train, y_test)
                print("fitness in DA:", fitness)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = positions[i].copy()

                inertia = self.w * velocities[i]
                cognitive = self.c * np.random.uniform(0, 1, self.n_features) * (best_solution - positions[i])
                social = self.s * np.random.uniform(0, 1, self.n_features) * (positions.mean(axis=0) - positions[i])
                velocities[i] = inertia + cognitive + social
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], 0, 1)

        return np.where(best_solution >= 0.5)[0]


class ArtificialBeeColony:
    def __init__(self, n_features, n_bees=10, max_iterations=10, limit=5):
        self.n_features = n_features
        self.n_bees = n_bees
        self.max_iterations = max_iterations
        self.limit = limit
        print("Default values initialised for ABC algorithm")

    def evaluate_features(self, features, X_train, X_test, y_train, y_test):
        if len(features) == 0:
            return 0
        selected_X_train = X_train[:, features]
        selected_X_test = X_test[:, features]
        # Compute mutual information scores for selected features
        mi_scores = mutual_info_classif(selected_X_train, y_train)
        # Return the mean of mutual information scores as fitness value
        return np.mean(mi_scores)

    def optimize(self, X_train, X_test, y_train, y_test):
        print("In optimze function of ABC")
        positions = np.random.randint(0, 2, size=(self.n_bees, self.n_features)).astype(float)
        fitness = np.zeros(self.n_bees)
        trial = np.zeros(self.n_bees)
        best_solution = positions[0]
        best_fitness = 0.0

        for iteration in range(self.max_iterations):
            for i in range(self.n_bees):
                features = np.where(positions[i] >= 0.5)[0]
                fitness[i] = self.evaluate_features(features, X_train, X_test, y_train, y_test)
                if fitness[i] > best_fitness:
                    best_fitness = fitness[i]
                    best_solution = positions[i].copy()

            for i in range(self.n_bees):
                partner = np.random.randint(0, self.n_bees)
                candidate = np.copy(positions[i])
                phi = np.random.uniform(-1, 1, self.n_features)
                candidate += phi * (positions[i] - positions[partner])
                candidate = np.clip(candidate, 0, 1)
                candidate_fitness = self.evaluate_features(np.where(candidate >= 0.5)[0], X_train, X_test, y_train, y_test)
                if candidate_fitness > fitness[i]:
                    positions[i] = candidate
                    fitness[i] = candidate_fitness
                    trial[i] = 0
                else:
                    trial[i] += 1

            for i in range(self.n_bees):
                if trial[i] > self.limit:
                    positions[i] = np.random.randint(0, 2, self.n_features)
                    trial[i] = 0

        return np.where(best_solution >= 0.5)[0]


class DataPreprocessors:
    def __init__(self, data_path):
        self.data_path = data_path

    def preprocess_and_train(self, max_iterations_da=2, max_iterations_abc=2):
        df = pd.read_csv(self.data_path)
        print(df.head())
        df.dropna(inplace=True)
        df = pd.get_dummies(df, drop_first=True)
        if 'label' not in df.columns:
            raise ValueError("The dataset does not contain a 'label' column.")
        X = df.drop(['label'], axis=1)
        y = df['label']
        print("X:", X)
        print("y:", y)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        selected_features = self.hybrid_feature_selection(X_scaled, y, max_iterations_da, max_iterations_abc)

        if len(selected_features) == 0:
            raise ValueError("No features selected by the hybrid feature selection method.")
        X_selected = X_scaled[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
        selected_feature_names = X.columns[selected_features]
        print("Selected Features:", selected_feature_names)

    def hybrid_feature_selection(self, X, y, max_iterations_da, max_iterations_abc):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        da = DragonflyAlgorithm(n_features=X_train.shape[1], max_iterations=max_iterations_da)
        selected_features_da = da.optimize(X_train, X_test, y_train, y_test)
        abc = ArtificialBeeColony(n_features=X_train.shape[1], max_iterations=max_iterations_abc)
        selected_features_abc = abc.optimize(X_train, X_test, y_train, y_test)
        final_selected_features = np.intersect1d(selected_features_da, selected_features_abc)
        print("FINAL Selected Features:", final_selected_features)

        return final_selected_features


# Usage example
if __name__ == "__main__":
    data_path = 'HIVESHIELD_Project/dataset/dataset_sdn.csv'
    preprocessor = DataPreprocessors(data_path)
    preprocessor.preprocess_and_train(max_iterations_da=2, max_iterations_abc=2)
    print("Inside main function call")
