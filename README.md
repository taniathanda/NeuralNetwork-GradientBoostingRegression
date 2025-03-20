# NeuralNetwork-GradientBoostingRegression


# 1. Loading Libraries
# import numpy as np                                                  # library for numerical computing like arrays, mathematical operations, and data manipulation
# import pandas as pdas                                               # Pandas is used for working with tabular data (like CSV files) and performing data analysis.
# import matplotlib.pyplot as plt                                     # Matplotlib is a library for creating static, animated, and interactive visualizations.    
# import seaborn as sns                                               # Seaborn is a statistical data visualization library built on top of Matplotlib.
# from sklearn.model_selection import train_test_split                # train_test_split is used to split data into training and testing sets.
# from sklearn.preprocessing import StandardScaler, LabelEncoder      # StandardScalar: It normalizes features by removing the mean and scaling to unit variance. 
                                                                    # LabelEncoder: Converts categorical labels (e.g., "cat", "dog", "fish") into numeric values (e.g., 0, 1, 2).
# from sklearn.neural_network import MLPRegressor                     # MLPRegressor (Multi-Layer Perceptron Regressor) is a type of Neural Network for regression tasks.
                                                                    # It learns complex relationships between input features and output values.
# from sklearn.ensemble import GradientBoostingRegressor              # GradientBoostingRegressor (GBR) is an ensemble learning method that builds multiple decision trees sequentially. 
# from sklearn.metrics import mean_absolute_error, mean_squared_error # Measures the average absolute difference between actual and predicted values.Lower MAE means better accuracy.
# MSE =Similar to MAE but squares the errors before averaging.
# Penalizes larger errors more than MAE.


# 2. Data Preparation
    ## Load Dataset 
url = "C:\Tania\AI Fanshawe\Data Science\Assignment\Lab 4\Student_Performance.csv"
dataset = pdas.read_csv(url)
print(dataset.head())

    ## Handling Missing Values
dataset.dropna(inplace=True)

    ## Encoding Categorical Variables: Encode any categorical variables as required.
label_encoder = LabelEncoder()
dataset['Extracurricular Activities'] = label_encoder.fit_transform(dataset['Extracurricular Activities'])

    # Data Splitting
X = dataset.drop(columns=['Performance Index'])  # Target variable is 'Performance Index'
y = dataset['Performance Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Modeling:  
    # Neural Network Model: 
    ## Implement a neural network for regression on the training data.
neural_network_model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500, random_state=42)
    ## Explaination the model’s parameters and training process.
        ## hidden_layer_sizes = 100, one hidden layer with 100 neurons.
        ## activation='relu' , Specifies the activation function used in the hidden layers. 'relu' (Rectified Linear Unit) prevent gradient problem and allows the network to learn faster.
        ## solver='adam', Specifies the optimization algorithm used for weight updates. 'adam' (Adaptive Moment Estimation) adjusts learning rates dynamically and performs well on most datasets.
        ## max_iter=500, maximum number of training iterations 
        ## random_state=42, the model produces consistent results everytime when we run
        ## This MLP regressor will take in the input features (X_train), pass them through a hidden layer of 100 neurons, apply ReLU activation, and use Adam optimizer to adjust the weights iteratively for up to 500 epochs to minimize the error.    
neural_network_model.fit(X_train, y_train)
neural_network_predictions = neural_network_model.predict(X_test)
 
    # Gradient Boosting Model
        ## Implementing a gradient boosting model for regression on the training data. 
gradient_boosting_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        ## Describing the chosen parameters and training process.
            ## n_estimators=100 = number of decision trees
            ## learning_rate=0.1, number of step size 
            ## max_depth=3,  the depth of each decision tree to 3 levels.
            ## random_state=42, the model produces consistent results everytime when we run
gradient_boosting_model.fit(X_train, y_train)
gradient_boosting_predictions = gradient_boosting_model.predict(X_test)

# 4. Evaluation Metrics
    ## Metrics on Test Data: Calculating and displaying the following regression performance metrics on the test set for both models: 
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)   # Mean Absolute Error (MAE)
    mse = mean_squared_error(y_true, y_pred)    # Mean Squared Error (MSE)
    print(f"{model_name} Performance:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}\n") 
evaluate_model(y_test, neural_network_predictions, "Neural Network")
evaluate_model(y_test, gradient_boosting_predictions, "Gradient Boosting")

    ## Comparison & Conclusion : Compare the performance metrics of both models and discuss which model performed better on the test set and why. 
models = ["Neural Network", "Gradient Boosting"]
mae_values = [mean_absolute_error(y_test, neural_network_predictions), mean_absolute_error(y_test, gradient_boosting_predictions)]
mse_values = [mean_squared_error(y_test, neural_network_predictions), mean_squared_error(y_test, gradient_boosting_predictions)]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.barplot(x=models, y=mae_values)
plt.title("Mean Absolute Error (MAE)")

plt.subplot(1, 2, 2)
sns.barplot(x=models, y=mse_values)
plt.title("Mean Squared Error (MSE)")

plt.show()

 

