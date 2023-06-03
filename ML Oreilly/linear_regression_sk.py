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


########## TRAINING DATA ###########

#            Country  GDP per capita (USD)  Life satisfaction
# 0           Russia          26456.387938                5.8
# 1           Greece          27287.083401                5.4
# 2           Turkey          28384.987785                5.5
# 3           Latvia          29932.493910                5.9
# 4          Hungary          31007.768407                5.6
# 5         Portugal          32181.154537                5.4
# 6           Poland          32238.157259                6.1
# 7          Estonia          35638.421351                5.7
# 8            Spain          36215.447591                6.3
# 9         Slovenia          36547.738956                5.9
# 10       Lithuania          36732.034744                5.9
# 11          Israel          38341.307570                7.2
# 12           Italy          38992.148381                6.0
# 13  United Kingdom          41627.129269                6.8
# 14          France          42025.617373                6.5
# 15     New Zealand          42404.393738                7.3
# 16          Canada          45856.625626                7.4
# 17         Finland          47260.800458                7.6
# 18         Belgium          48210.033111                6.9
# 19       Australia          48697.837028                7.3
# 20          Sweden          50683.323510                7.3
# 21         Germany          50922.358023                7.0
# 22         Austria          51935.603862                7.1
# 23         Iceland          52279.728851                7.5
# 24     Netherlands          54209.563836                7.4
# 25         Denmark          55938.212809                7.6
# 26   United States          60235.728492                6.9