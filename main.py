import pandas as pd
import numpy as np
from sklearn import linear_model, metrics, model_selection,datasets,tree
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# Load the iris dataset
x,y=datasets.load_iris(return_X_y=True)
#print(x.shape) # (150, 4)
#print(y.shape) # (150,)

# EDA
df = pd.DataFrame(x, columns=datasets.load_iris().feature_names)
df['species'] = y

sns.pairplot(df, hue='species')
plt.show()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.1, random_state=42,stratify=y
    )

# Create a logistic regression model
model = linear_model.LogisticRegression(max_iter=200)

# Fit the model to the training data
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}") # accuracy of the logistic regression model = .933

# Generate the confusion matrix
cm=metrics.confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

# Decision Tree Classifier
dt_model = tree.DecisionTreeClassifier()
dt_model.fit(x_train, y_train)
dt_y_pred = dt_model.predict(x_test)
dt_accuracy = metrics.accuracy_score(y_test, dt_y_pred)
print(f"Decision Tree Accuracy: {dt_accuracy}") # accuracy of the decision tree model = .867

# Save the logistic regression model
with open('logistic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as logistic_model.pkl")
    

