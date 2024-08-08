#importing required packages
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history, plot_contour

#Dragonfly algorithm 
class DragonflyAlgorithm:
    def __init__(self, n_features, n_dragonflies=20, max_iterations=15, w=0.9, c=0.3, s=0.3):
        print("Executing Dragonfly Optimization Algorithm")
        self.n_features = n_features
        self.n_dragonflies = n_dragonflies
        self.max_iterations = max_iterations
        self.w = w
        self.c = c
        self.s = s
        self.best_fitness_over_time = []
        self.cost_history = [] #initialize cost history
        
        
        print("Default values initialised for dragonfly algorithm....")

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
        print("Executing optimize function of DA algorithm")
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
                    
                self.cost_history.append(best_fitness)  # Save cost value

                inertia = self.w * velocities[i]
                cognitive = self.c * np.random.uniform(0, 1, self.n_features) * (best_solution - positions[i])
                social = self.s * np.random.uniform(0, 1, self.n_features) * (positions.mean(axis=0) - positions[i])
                velocities[i] = inertia + cognitive + social
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], 0, 1)
                print(best_solution)
            self.best_fitness_over_time.append(best_fitness)

        return np.where(best_solution >= 0.5)[0]
    def plot_fitness_over_time(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.best_fitness_over_time) + 1), self.best_fitness_over_time, marker='o', linestyle='-', color='b')
        plt.title('Best Fitness Value Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness Value')
        plt.grid(True)
        plt.show()
        plt.plot(self.cost_history)#plot of da cost history

        

#Artificial Bee Colony Algorithm
class ArtificialBeeColony:
    def __init__(self, n_features, n_bees=20, max_iterations=15, limit=5):
        print("Executing Artificial Bee Colony Optimization Algorithm")
        self.n_features = n_features
        self.n_bees = n_bees
        self.max_iterations = max_iterations
        self.limit = limit
        self.best_fitness_over_time=[]
        self.cost_history = []#initialize abc cost history
    
        print("Default values initialised for ABC algorithm.....")

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
        print("Executing optimize function of ABC algorithm")
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
                    print("best solution in ABC",best_solution)
                self.cost_history.append(best_fitness)  # Save cost value
            self.best_fitness_over_time.append(best_fitness)

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
    def plot_fitness_over_time(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.best_fitness_over_time) + 1), self.best_fitness_over_time, marker='o', linestyle='-', color='b')
        plt.title('Best Fitness Value Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness Value')
        plt.grid(True)
        plt.show()
        plt.plot(self.cost_history) #plot cost history of abc
        


#data preprocessing and hybrid feature selection  
class DataPreprocessors:
    def __init__(self, data_path):
        self.data_path = data_path
        self.da_cost_history = []  # Initialize cost history for Dragonfly Algorithm
        self.abc_cost_history = []  # Initialize cost history for Artificial Bee Colony
        

    def preprocess_and_train(self, max_iterations_da=15, max_iterations_abc=15):
        print("Exploratory Data Analysis on the DDoS Attack 2019 Dataset through visualization and Data Preprocessing....")
        df = pd.read_csv(self.data_path)
        print(df.shape)        
        print(df.columns)
        print(df.isnull().sum())
            
        df.dropna(inplace=True)
        df = pd.get_dummies(df, drop_first=True)
        #sns.heatmap(df.corr(),annot = True,cmap = 'terrain')
        #sns.pairplot(data=df)
        
        print(df.isnull().sum())
        print("This Dataframe has {} rows and {} columns after removing null values".format(df.shape[0], df.shape[1]))
        
        
        #selects rows from the DataFrame df where the value in the 'label' column is equal to 0 and 1 (indicating a normal or DDOS attack).
        malign = df[df['label'] == 1]
        benign = df[df['label'] == 0]

        #len calculates the number of rows where 'label' is 0 or 1 
        #calculates the percentage of DDOS attacks and normal flows relative to the total dataset size.
        print('Number of DDOS attacks that has occured :',round((len(malign)/df.shape[0])*100,2),'%')
        print('Number of DDOS attacks that has not occured :',round((len(benign)/df.shape[0])*100,2),'%')


        # Let's plot the Label class against the Frequency
        labels = ['benign','malign']
        # computes the percentage distribution of each label class
        #classes = pd.value_counts(df['label'], sort = True) / df['label'].count() *100
        classes = df['label'].value_counts(sort=True) / df['label'].count() * 100

        classes.plot(kind = 'bar') #creates a line plot of the label class distribution
        plt.title("Label class distribution")
        plt.xticks(range(2), labels)
        plt.xlabel("Label")
        plt.ylabel("Frequency %")
        
        
        #visualizing the distribution of a specific variable ('pktcount') across different categories defined by the 'label' column in the DataFrame. 
        #It leverages density plots to show the shape and spread of the data within each category#sets the transparency level of the density plot
        #import matplotlib.pyplot as plt

        # Assuming 'df' is your DataFrame containing the data
        # Assuming 'label' is the column you want to use for coloring

        # Set the style of the plot
        #plt.style.use('seaborn-whitegrid')
        #import seaborn as sns
        sns.set_style('whitegrid')


        # Plot density plots for each label category
        plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
        for label_category in df['label'].unique():
            df[df['label'] == label_category]['pktcount'].plot.density(label=label_category, alpha=0.5)
            # You can replace 'pktcount' with 'flows', 'bytecount', etc. to plot other variables

        # Set plot title and labels
        plt.title('Density Plot for Variables by Label')
        plt.xlabel('Values')
        plt.ylabel('Density')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()
        #sns.pairplot(df,hue="label",vars=['pktcount','flows','bytecount'])
        
        
        numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
        print("The number of numerical features is",len(numerical_features),"and they are : \n",numerical_features)
        
        categorical_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
        print("The number of categorical features is",len(categorical_features),"and they are : \n",categorical_features)
        
        #discrete numerical features 
        discrete_feature = [feature for feature in numerical_features if df[feature].nunique()<=15 and feature != 'label']
        print("The number of discrete features is",len(discrete_feature),"and they are : \n",discrete_feature)
        
        
        def countplot_distribution(col):
            sns.set_theme(style="darkgrid")
            sns.countplot(y=col, data=df).set(title = 'Distribution of ' + col)

        def histplot_distribution(col):
            sns.set_theme(style="darkgrid")
            sns.histplot(data=df,x=col, kde=True,color="red").set(title = 'Distribution of ' + col)
        ## Lets analyse the categorical values by creating histograms to understand the distribution
        f = plt.figure(figsize=(8,20))
        for i in range(len(categorical_features)):
            f.add_subplot(len(categorical_features), 1, i+1)
            countplot_distribution(categorical_features[i])
        plt.show()
        
        
        ## Lets analyse the continuous values by creating histograms to understand the distribution
        
        '''continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature + ['label']]
        print("The number of continuous_feature features is",len(continuous_feature),"and they are : \n",continuous_feature)
        f = plt.figure(figsize=(20,90))
        for i in range(len(continuous_feature)):
            f.add_subplot(len(continuous_feature), 2, i+1)
            histplot_distribution(continuous_feature[i])
        plt.show()'''
        
        #numerical discrete features
        '''for feature in discrete_feature:
            plt.figure(figsize=(8,4))
            cat_num = df[feature].value_counts()
            sns.barplot(x=cat_num.index, y = cat_num).set(title = "Graph for "+feature, ylabel="Frequency")
            plt.show()'''
        
        #heatmap of correlation of features
        
        correlation_matrix = df.corr()
        fig = plt.figure(figsize=(17,17))
        mask = np.zeros_like(correlation_matrix, dtype=np.bool)
        mask[np.triu_indices_from(mask)]= True
        sns.set_theme(style="darkgrid")
        ax = sns.heatmap(correlation_matrix,square = True,annot=True,center=0,vmin=-1,linewidths = .5,annot_kws = {"size": 11},mask = mask)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45, horizontalalignment='right');
        plt.show()
        
        if 'label' not in df.columns:
            raise ValueError("The dataset does not contain a 'label' column.")
        X = df.drop(['label'], axis=1)
        y = df['label']
        print("X:", X)
        print("y:", y)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        print("Succesfully completed the required EDA and Preprocessing.")
        selected_features = self.hybrid_feature_selection(X_scaled, y, max_iterations_da, max_iterations_abc)

        if len(selected_features) == 0:
            raise ValueError("No features selected by the hybrid feature selection method.")
        X_selected = X_scaled[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
        selected_feature_names = X.columns[selected_features]
        print("Selected Features:", selected_feature_names)
        
        print("MOHADA Feature Selection Module executed successfully")
        
        
        # Save the preprocessed data
        np.save('X_train.npy', X_train)
        np.save('X_test.npy', X_test)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)
        subprocess.run(['python', 'classifier-svm.py'], check=True)
        # Visualize optimization history
        self.visualize_optimization_history()
        print("_______________________________________________________")
        

    def hybrid_feature_selection(self, X, y, max_iterations_da, max_iterations_abc):
        print("Moving on to the proposed HYBRID Feature Selection Module.....")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        
        #DA
        da = DragonflyAlgorithm(n_features=X_train.shape[1], max_iterations=max_iterations_da)
        selected_features_da = da.optimize(X_train, X_test, y_train, y_test)
        da.plot_fitness_over_time()
        self.da_cost_history = da.cost_history  # Store cost history
        
        
        # Save the data for MATLAB
        with open('dragonfly_solution_history.pkl', 'wb') as file:
            pickle.dump(da.cost_history, file)

        with open('dragonfly_fitness_history.pkl', 'wb') as file:
            pickle.dump(da.best_fitness_over_time, file)
        
        print("Successfully executed DA....")
        print("Selected features resulted after execution of DA:",selected_features_da)
        
        #ABC implementation
        abc = ArtificialBeeColony(n_features=X_train.shape[1], max_iterations=max_iterations_abc)
        selected_features_abc = abc.optimize(X_train, X_test, y_train, y_test)
        abc.plot_fitness_over_time()
        self.abc_cost_history = abc.cost_history  # Store cost history
        
        
        # Save the data for MATLAB
        with open('abc_cost_history.pkl', 'wb') as file:
            pickle.dump(abc.cost_history, file)

        with open('abc_fitness_history.pkl', 'wb') as file:
            pickle.dump(abc.best_fitness_over_time, file)
        
        
        print("Successfully executed ABC....")
        print("Selected features resulted after execution of ABC:",selected_features_abc)
        
        #MOHADA
        final_selected_features = np.intersect1d(selected_features_da, selected_features_abc)
        print("Combining results rendered by DA and ABC.....")
        print("FEATURES EXTRACTED BY MOHADA ALGORITHM:", final_selected_features)
        self.visualize_optimization_history()
        
        # Plot fitness values for DA
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(da.best_fitness_over_time) + 1), da.best_fitness_over_time, marker='o', linestyle='-', color='b')
        plt.title('Best Fitness Value Over Iterations (DA)')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness Value')
        plt.grid(True)

        # Plot fitness values for ABC
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(abc.best_fitness_over_time) + 1), abc.best_fitness_over_time, marker='o', linestyle='-', color='r')
        plt.title('Best Fitness Value Over Iterations (ABC)')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness Value')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=final_selected_features)
        plt.title('Box Plot of Final Selected Features')
        plt.xlabel('Features')
        plt.ylabel('Value')
        plt.show()

        return final_selected_features
    def visualize_optimization_history(self):
        plt.figure(figsize=(12, 6))

        # Plot DA cost history
        plt.plot(self.da_cost_history, label='Dragonfly Algorithm Cost History')
        # Plot ABC cost history
        plt.plot(self.abc_cost_history, label='Artificial Bee Colony Cost History')

        plt.xlabel('Iteration')
        plt.ylabel('Cost Fitness Function)')
        plt.title('Optimization Cost History')
        plt.legend()
        plt.show()

import pandas as pd
# Usage example
if __name__ == "__main__":
    data_path = 'dataset\dataset_sdn.csv'
    df = pd.read_csv(data_path)
    print("***HIVESHIELD - SWARM INTELLIGENCE FOR DDoS ATTACK DETECTION***")
    print("_______________________________________________________________")
    preprocessor = DataPreprocessors(data_path)
    print("Executing the Hybrid feature selection module...")
    preprocessor.preprocess_and_train(max_iterations_da=15, max_iterations_abc=15)
    



    

        
        


