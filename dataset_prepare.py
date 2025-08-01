import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

# Load Dataset
df = pd.read_excel("Online Retail.xlsx")
print("📦 Initial dataset shape:", df.shape)

# Remove duplicate rows first
initial_shape = df.shape
df = df.drop_duplicates()
print(f"🧹 Removed {initial_shape[0] - df.shape[0]} duplicate rows. New shape: {df.shape}")

# Define non-product StockCodes (update this list as needed)
non_product_stockcodes = ['SAMPLE', 'POST','MANUAL']

# Create issue flags
df['MissingCustomerID'] = df['CustomerID'].isnull()
df['CanceledInvoice'] = df['InvoiceNo'].astype(str).str.startswith('C')
df['NonPositiveQuantity'] = df['Quantity'] <= 0
df['NonPositiveUnitPrice'] = df['UnitPrice'] <= 0
df['MissingDescription'] = df['Description'].isnull()
df['NonProductStockCode'] = df['Description'].isnull() | df['Description'].str.strip().eq('')

# Create combined issue label for analysis
def issue_label(row):
    flags = []
    if not row['MissingCustomerID']:
        flags.append('Clean')
    else:
        flags.append('MissingCustomerID')
    if row['CanceledInvoice']:
        flags.append('CanceledInvoice')
    if row['NonPositiveQuantity']:
        flags.append('NonPositiveQuantity')
    if row['NonPositiveUnitPrice']:
        flags.append('NonPositiveUnitPrice')
    if row['MissingDescription']:
        flags.append('MissingDescription')
    if row['NonProductStockCode']:
        flags.append('NonProductStockCode')
    return '+'.join(flags)

df['IssueCombination'] = df.apply(issue_label, axis=1)

# Print data quality issue summary (informational)
issue_counts = df['IssueCombination'].value_counts()
print("\n⚠️ Data Quality Issue Combinations and Affected Rows:\n")
print(issue_counts)

print("\n📋 Individual Issue Counts:")
print(f"Missing CustomerID: {df['MissingCustomerID'].sum()}")
print(f"Canceled Invoice: {df['CanceledInvoice'].sum()}")
print(f"Non-positive Quantity: {df['NonPositiveQuantity'].sum()}")
print(f"Non-positive UnitPrice: {df['NonPositiveUnitPrice'].sum()}")
print(f"Missing Description: {df['MissingDescription'].sum()}")
print(f"Non-product StockCode: {df['NonProductStockCode'].sum()}")

# Remove rows with missing CustomerID OR non-product StockCode OR
# combinations involving MissingCustomerID + NonPositiveQuantity + NonPositiveUnitPrice + MissingDescription
# and MissingCustomerID + NonPositiveUnitPrice + MissingDescription
# **But do NOT remove canceled invoices here, keep them for matched cancellation removal**
df_clean = df[~(
    df['MissingCustomerID'] |
    df['NonProductStockCode'] |
    (df['MissingCustomerID'] & df['NonPositiveQuantity'] & df['NonPositiveUnitPrice'] & df['MissingDescription']) |
    (df['MissingCustomerID'] & df['NonPositiveUnitPrice'] & df['MissingDescription'])
)]

print(f"\n✅ After initial removal (excluding canceled invoices), shape: {df_clean.shape}")

# Function to remove matched cancellations and purchases
def remove_matched_cancellations(df):
    df = df.copy()
    df['IsCancelled'] = df['InvoiceNo'].astype(str).str.startswith('C')

    cancelled = df[df['IsCancelled']]
    purchases = df[~df['IsCancelled']].copy()
    purchases['Matched'] = False

    to_remove = []

    for idx, row in cancelled.iterrows():
        cust = row['CustomerID']
        stock = row['StockCode']
        qty = abs(row['Quantity'])
        price = row['UnitPrice']

        match = purchases[
            (purchases['CustomerID'] == cust) &
            (purchases['StockCode'] == stock) &
            (purchases['Quantity'] == qty) &
            (purchases['UnitPrice'] == price) &
            (~purchases['Matched'])
        ]

        if not match.empty:
            match_idx = match.index[0]
            purchases.at[match_idx, 'Matched'] = True
            to_remove.extend([idx, match_idx])

    df_cleaned = df.drop(index=to_remove)
    df_cleaned.drop(columns=['IsCancelled'], inplace=True)
    print(f"✅ Removed {len(to_remove)//2} matched cancellations and purchases.")
    return df_cleaned

# Remove matched cancellations and their purchases
df_clean = remove_matched_cancellations(df_clean)

# Remove any remaining canceled invoices and invalid Quantity or UnitPrice
df_clean = df_clean[
    (~df_clean['InvoiceNo'].astype(str).str.startswith('C')) &
    (df_clean['Quantity'] > 0) &
    (df_clean['UnitPrice'] > 0)
]

print(f"\n✅ Final cleaned dataset shape after removing leftover canceled invoices and invalid quantities/prices: {df_clean.shape}")

# Drop auxiliary columns no longer needed
df_clean = df_clean.drop(columns=[
    'MissingCustomerID', 'CanceledInvoice', 'NonPositiveQuantity',
    'NonPositiveUnitPrice', 'MissingDescription', 'NonProductStockCode', 'IssueCombination'
])

# Confirm datetime parsing
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

# Statistical Summary AFTER cleaning
print("\n📊 Statistical Summary (After Cleaning):\n", df_clean[['Quantity', 'UnitPrice']].describe())

# Visualizations AFTER cleaning
plt.figure(figsize=(18, 4))
sns.histplot(np.log1p(df_clean['Quantity']), bins=50, kde=True)
plt.title('Log-Scaled Quantity Distribution (After Cleaning)')
plt.xlabel('log(1 + Quantity)')
plt.show()



# Boxplots AFTER cleaning
plt.figure(figsize=(18, 4))
sns.boxplot(x=df_clean['Quantity'])
plt.title('Boxplot: Quantity (After Cleaning)')
plt.xlim(0, 100000)
plt.xticks(np.arange(10000, 100001, 10000))
plt.show()



# Top 5 countries by transaction count
top_countries = df_clean['Country'].value_counts().head(5)
print("\n🌍 Top 5 Countries by Number of Transactions:\n", top_countries)

# Prepare final dataframe for Apriori (drop InvoiceDate if not needed)
df_final = df_clean.drop(columns=['InvoiceDate'])
print("\n📁 Final dataframe ready for Apriori:")
print(df_final.head())

# Save cleaned dataset
df_clean.to_excel("Cleaned_Online_Retail.xlsx", index=False)
print("\n✅ Cleaned dataset saved as 'Cleaned_Online_Retail.xlsx'")
