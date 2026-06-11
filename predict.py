import pickle
# Predict using the saved model
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example input (replace with actual feature values)
print("Enter the features of the flower.")
sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

# Make predictions on new data
species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
new_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(new_data)
print(f"Prediction: {species_mapping[prediction[0]]}")