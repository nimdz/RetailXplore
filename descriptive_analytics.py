import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker


# Load cleaned dataset
df = pd.read_excel('Cleaned_Online_Retail.xlsx')

# --- Basic Descriptive Analytics ---
unique_customers = df['CustomerID'].nunique()
unique_invoices = df['InvoiceNo'].nunique()
total_revenue = (df['Quantity'] * df['UnitPrice']).sum()
num_countries = df['Country'].nunique()

print(f"Unique Customers: {unique_customers}")
print(f"Unique Transactions (Invoices): {unique_invoices}")
print(f"Total Revenue: £{total_revenue:,.2f}")
print(f"Number of Countries: {num_countries}")

# --- Top 10 Most Frequently Purchased Products by Quantity ---
top_products = (
    df.groupby(['StockCode', 'Description'])['Quantity']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
print("\nTop 10 Most Frequently Purchased Products:")
print(top_products)

# --- Monthly Sales Trends ---

# Convert InvoiceDate to datetime if needed
if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')

monthly_sales = (
    df.groupby('InvoiceMonth')
    .agg(
        total_quantity=('Quantity', 'sum'),
        total_revenue=('Quantity', lambda x: (x * df.loc[x.index, 'UnitPrice']).sum()),
        transactions=('InvoiceNo', 'nunique')
    )
    .reset_index()
)
monthly_sales['InvoiceMonth'] = monthly_sales['InvoiceMonth'].dt.to_timestamp()

# Plot monthly sales quantity and revenue
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.bar(monthly_sales['InvoiceMonth'], monthly_sales['total_quantity'], color='skyblue', alpha=0.6, label='Total Quantity Sold')
ax1.set_ylabel('Total Quantity Sold', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')

ax2 = ax1.twinx()
ax2.plot(monthly_sales['InvoiceMonth'], monthly_sales['total_revenue'], color='darkgreen', label='Total Revenue (£)')
ax2.set_ylabel('Total Revenue (£)', color='darkgreen')
ax2.tick_params(axis='y', labelcolor='darkgreen')

plt.title('Monthly Sales Trends')
fig.tight_layout()
plt.show()

# --- Country Level Analysis ---
country_summary = (
    df.groupby('Country')
    .agg(
        total_quantity=('Quantity', 'sum'),
        total_revenue=('Quantity', lambda x: (x * df.loc[x.index, 'UnitPrice']).sum()),
        unique_customers=('CustomerID', 'nunique'),
        transactions=('InvoiceNo', 'nunique')
    )
    .sort_values(by='total_quantity', ascending=False)
    .reset_index()
)
print("\nCountry Level Summary:")
print(country_summary.head(10))

# --- Correlation Matrix of Numeric Variables ---

numeric_cols = ['Quantity', 'UnitPrice']
# Add TotalPrice for correlation
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

numeric_cols.append('TotalPrice')

corr = df[numeric_cols].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Variables')
plt.show()


# Plot: Top 10 Most Frequently Purchased Products (Quantity)
plt.figure(figsize=(12,6))
sns.barplot(
    data=top_products,
    x='Quantity',
    y='Description',
    palette='viridis'
)
plt.title('Top 10 Most Frequently Purchased Products (Quantity Sold)')
plt.xlabel('Quantity Sold')
plt.ylabel('Product Description')
plt.tight_layout()
plt.show()

# Prepare top 10 countries by total_quantity
top_countries = country_summary.head(10)

# Plot: Top 10 Countries by Total Quantity Sold
plt.figure(figsize=(12,6))
sns.barplot(
    data=top_countries,
    x='total_quantity',
    y='Country',
    palette='magma'
)
plt.title('Top 10 Countries by Total Quantity Sold')
plt.xlabel('Total Quantity Sold')
plt.ylabel('Country')
plt.tight_layout()
plt.show()
