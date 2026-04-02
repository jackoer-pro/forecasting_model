import pandas as pd
import matplotlib.pyplot as plt
orders=pd.read_csv("data/orders.csv")
product=pd.read_csv("data/products.csv")
order_product=pd.read_csv("data/order_products__prior.csv")
print(orders.head())
print("########################################################################################")
print(product.head())
print("########################################################################################")
print(order_product.head())
merged=orders.merge(order_product, on="order_id")
merged=merged.merge(product, on="product_id")
# merge the table
merged = merged[['user_id', 'order_id', 'order_number', 'product_id', 'product_name', 'add_to_cart_order', 'reordered', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']]
print(merged.head())
print("########################################################################################")
# Investigate how many unique products as well as customers
print("Unique users:", merged['user_id'].nunique())
print("Unique products:", merged['product_id'].nunique())
print("########################################################################################")
# How many orders per customer
orders_per_customer=merged.groupby('user_id')['order_id'].nunique().sort_values(ascending=False)
print(orders_per_customer.head())
print(orders_per_customer.describe())
# Overall reorder rate
reorder_rate = merged['reordered'].mean()
print("Overall reorder rate:", reorder_rate)
# top 10 best seller with the top contributor of them
top_seller=merged['product_name'].value_counts().head(10).index
for top in top_seller:
    product_data=merged[merged['product_name']== top]
    top_customer=product_data['user_id'].value_counts().idxmax()
    count=product_data['user_id'].value_counts().max()
    print(f"{top}: Top customer {top_customer} bought {count} times")
# investigate the order per day to find the salience
merged['order_dow'].value_counts().sort_index().plot(kind="bar")
plt.title("Orders by day of the week")
plt.xlabel("Day of the week (0-Sunday)")
plt.ylabel("Number of Orders")
plt.show()
# top 10 top product with their reorder time and percentage 
product_stats=merged.groupby('product_name').agg(
    total_orders=("order_id","count"),
    total_reorder=("reordered","sum")
)
product_stats['reorder_percentage']=product_stats['total_reorder']/product_stats['total_orders']
product_stats=product_stats.sort_values('total_orders',ascending=False)
print(product_stats.head(10))