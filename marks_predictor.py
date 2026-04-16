import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [20, 25, 35, 40, 50, 55, 65, 70]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

try:
    hours = float(input("Enter study hours: "))
    predicted_marks = model.predict([[hours]])
    print(f"Expected Marks: {predicted_marks[0]:.2f}")
except ValueError:
    print("Please enter a valid number!")

plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")

plt.show()