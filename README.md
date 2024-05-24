# House_Price_Prediction
House Price Prediction is a common machine learning task that involves building a model to estimate the price of houses based on various features or attributes. In this task, we use a linear regression model to make predictions. Linear regression is a simple and widely used method for predicting a continuous target variable, such as the price of a house, based on one or more input features like square footage, the number of bedrooms, and the number of bathrooms.

Here's a description of the code for House Price Prediction using Linear Regression:

1. **Importing Necessary Libraries**: In this code, we start by importing the required Python libraries. These libraries include NumPy for numerical operations, Pandas for data manipulation, Matplotlib for data visualization, and scikit-learn for machine learning.

2. **Loading the Dataset**: The code assumes that you have a dataset in a CSV file. You need to replace `'data.csv'` with the actual path to your dataset. The dataset typically includes information about houses, such as square footage, the number of bedrooms, the number of bathrooms, and the corresponding prices.

3. **Data Splitting**: The dataset is split into training and testing sets using the `train_test_split` function from scikit-learn. The training set is used to train the linear regression model, and the testing set is used to evaluate the model's performance.

4. **Creating a Linear Regression Model**: A Linear Regression model is created using the `LinearRegression` class from scikit-learn.

5. **Training the Model**: The model is trained on the training data using the `fit` method. It learns to predict house prices based on the provided features.

6. **Making Predictions**: After training, the model is used to make predictions on the test data using the `predict` method.

7. **Model Evaluation**: The code calculates two important evaluation metrics:
   - **Mean Squared Error (MSE)**: This metric measures the average squared difference between predicted and actual prices. A lower MSE indicates a better model.
   - **R-squared (R2)**: This metric measures the goodness of fit of the model. It ranges from 0 to 1, with 1 indicating a perfect fit.

8. **Visualizing the Results**: The code creates a scatter plot to visualize the relationship between actual and predicted house prices. This provides a visual representation of how well the model performs.

To use this code, you'll need to replace `'data.csv'` with the path to your own dataset containing information about houses and their prices. Make sure your dataset has the appropriate columns and features for this task, and ensure you have the necessary Python libraries installed in your environment. Once you've done that, you can run the code in a Jupyter Notebook or another Python environment to predict house prices based on the provided features.
