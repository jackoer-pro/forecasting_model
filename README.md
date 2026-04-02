## Dataset
The project uses the Instacart dataset from Kaggle:

| File | Description |
|------|-------------|
| `orders.csv` | Information about all orders |
| `products.csv` | Product details including name and category |
| `order_products__prior.csv` | Products in users’ previous orders |
| `order_products__train.csv` | Products in most recent orders |
| `aisles.csv` | Product aisle information |
| `departments.csv` | Product department information |

[Dataset source](https://www.kaggle.com/c/instacart-market-basket-analysis/data)

---

## Tools & Libraries
- **Python** – data processing and analysis  
- **Pandas, NumPy** – data manipulation  
- **Matplotlib** – visualization    
- **Scikit-learn** – Linear Regression for forecasting    

---

## Project Steps

### 1. Data Loading & Cleaning
- Merge datasets to create a full order-product table  
- Handle missing values and duplicates  

### 2. Exploratory Data Analysis (EDA)
- Identify top-selling products and categories  
- Analyze most active users and order patterns  
- Visualize order frequency over time  

### 3. Market Basket Analysis
- Generate frequent itemsets  
- Create association rules to find products often bought together  
- Filter rules by confidence, support, and lift  

### 4. Sales Forecasting
- Aggregate historical sales data per product  
- Apply **Linear Regression** to predict future demand  
- Visualize forecast trends with actual vs predicted sales  

### 5. Insights & Recommendations
- Suggest product bundles based on association rules  
- Recommend products for marketing campaigns  
- Forecast inventory needs for better stock management  

---
