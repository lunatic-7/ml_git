import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Download and prepare the data
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
print(lifesat)  # printing csv (bss dkhne k ly)

X = lifesat[["GDP per capita (USD)"]].values
Y = lifesat[["Life satisfaction"]].values

# Visualize the data
lifesat.plot(kind="scatter", grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500,62_500,4,9])  # x_start=23_500, x_end=62_500, y_start=4, y_end=9
plt.show()

# Select a Linear model
model = LinearRegression()

# Train the model
model.fit(X, Y)

# Make a prediction for Cyprus
X_new = [[37_655.2]]  # Cyprus GDT per capita in 2020
print(model.predict(X_new))  # Output: [[6.30165767]]