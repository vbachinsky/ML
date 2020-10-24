import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as linear_model


advertising_data = pd.read_csv('data/advertising.csv', index_col=0)

tv_data = advertising_data[['TV']]
radio_data = advertising_data[['radio']]
nwsp_data = advertising_data[['newspaper']]
sales_data = advertising_data[["sales"]]

# Calculate correlation
tv_vs_sales = tv_data.join(sales_data)
radio_vs_sales = radio_data.join(sales_data)
nwsp_vs_sales = nwsp_data.join(sales_data)

print('Correlation for TV and sales: {}'.format(tv_vs_sales.corr()))
print('Correlation for radio and sales: {}'.format(radio_vs_sales.corr()))
print('Correlation for newspaper and sales: {}'.format(nwsp_vs_sales.corr()))

# Calculate regression models
linear_regression_tv = linear_model.LinearRegression()
linear_regression_radio = linear_model.LinearRegression()
linear_regression_newspaper = linear_model.LinearRegression()

linear_regression_tv.fit(X=tv_data, y=sales_data)
linear_regression_radio.fit(X=radio_data, y=sales_data)
linear_regression_newspaper.fit(X=nwsp_data, y=sales_data)

regression_line_points_tv = linear_regression_tv.predict(X=tv_data)
regression_line_points_radio = linear_regression_radio.predict(X=radio_data)
regression_line_points_nwsp = linear_regression_newspaper.predict(X=nwsp_data)

print(linear_regression_tv.coef_)
print(linear_regression_radio.coef_)
print(linear_regression_newspaper.coef_)

plt.plot(regression_line_points_tv, tv_data,
         regression_line_points_radio, radio_data,
         regression_line_points_nwsp, nwsp_data)

plt.show()
