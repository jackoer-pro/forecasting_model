import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

orders=pd.read_csv("data/orders.csv")
product=pd.read_csv("data/products.csv")
order_product=pd.read_csv("data/order_products__prior.csv")
merged=orders.merge(order_product, on="order_id")
merged=merged.merge(product, on="product_id")
merged = merged[['user_id', 'order_id', 'order_number', 'product_id', 'product_name', 'add_to_cart_order', 'reordered', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']]
merged['days_since_last_order']=merged.groupby('user_id')['days_since_prior_order'].cumsum().fillna(0)
merged['week']=(merged['days_since_last_order']//7).astype(int)
# we want to view the consumption of one product for each week and reorder of that product
weekly_product_demand=merged.groupby(['product_id','week']).agg(
    total_order=('order_id','count'),
    total_reorder=('reordered','sum')
).reset_index()
weekly_product_demand=weekly_product_demand.merge(product[['product_id','product_name']],on='product_id')
# sort the value to gurantee the precision
weekly_product_demand=weekly_product_demand.sort_values(['product_id','week'])
# create new columns to show previous data along side with present (create a model to look at the past to predict the future)
# lag_1 used capture immediate trend, lag_2 used to capture slightly older trend to maintain stability
weekly_product_demand["lag_1"]=(weekly_product_demand.groupby('product_id')['total_order'].shift(1))
weekly_product_demand["lag_2"]=(weekly_product_demand.groupby('product_id')['total_order'].shift(2))
# come up with more stable columns to predict trend (mean of 4 weeks) to prevent unpredictable week
# rolling_4 captures overal trend, reduces noise
weekly_product_demand['rolling_4']=(weekly_product_demand.groupby('product_id')['total_order'].rolling(4).mean().reset_index(0,drop=True))
# handle some rows with missing values (so we will delete frist few rows because the model can not operate properly with missing inputs)
weekly_product_demand = weekly_product_demand.dropna()
# Train & test
# The first 50 weeks was fed to the model to detect the pattern then predict future
# The next 50 ones used to test the precision and deviation of the model
train = weekly_product_demand[weekly_product_demand['week'] < 50]
test = weekly_product_demand[weekly_product_demand['week'] >= 50]
# feed in the model to let it know what is the inputs and what is the outputs
X_train = train[['lag_1','lag_2','rolling_4']]
y_train = train['total_order']
#Train model
model = LinearRegression()
model.fit(X_train, y_train)
#predict
X_test = test[['lag_1','lag_2','rolling_4']]
y_test = test['total_order']
preds = model.predict(X_test)
#evaluate
mae = mean_absolute_error(y_test, preds)
print("Mean Absolute Error:", mae)
# visualise the predicted data along side with real data
results = test.copy()
results['predicted'] = preds

# Show table for first 10 rows
print(results[['product_id','week','total_order','predicted']].head(10))
# Filter only for this product
product_id = 1
sample = results[results['product_id'] == product_id].sort_values('week').reset_index(drop=True)

# Filter weeks 50 to 60
sample = sample[(sample['week'] >= 50) & (sample['week'] <= 60)].reset_index(drop=True)

x = np.arange(len(sample))
width = 0.35


plt.figure(figsize=(12,6))
plt.bar(x - width/2, sample['total_order'], width, label='Actual')
plt.bar(x + width/2, sample['predicted'], width, label='Predicted')

plt.xticks(x, sample['week'])  # show real week numbers on x-axis
plt.xlabel('Week')
plt.ylabel('Total Orders')
plt.title(f'Product {product_id} - Actual vs Predicted')
plt.legend()
plt.show()