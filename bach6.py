import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as linear_model

subs_from_ads = pd.read_csv('data/subscribers_from_ads.csv')

print(subs_from_ads)

plt.scatter(subs_from_ads[["promotion_budget"]], subs_from_ads[["subscribers"]])

promotion_budget = subs_from_ads[["promotion_budget"]]
number_of_subscribers = subs_from_ads[["subscribers"]]

linear_regression = linear_model.LinearRegression()
linear_regression.fit(X=promotion_budget, y=number_of_subscribers)  # X - big, y - small

print(linear_regression.coef_)
print(linear_regression.intercept_)

regression_line_points = linear_regression.predict(X=promotion_budget)
plt.plot(promotion_budget, regression_line_points)

plt.show()
