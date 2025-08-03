import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load cleaned dataset
df_cleaned = pd.read_excel("Cleaned_Online_Retail.xlsx")
print("📦 Initial dataset shape:", df_cleaned.shape)

# Top 3 countries with highest number of transactions
top_countries = df_cleaned['Country'].value_counts().head(3).index.tolist()
print(f"🌍 Top 3 Countries selected for Basket Analysis: {top_countries}")

# Function to prepare basket for Apriori
def prepare_basket(df_country):
    basket = (df_country
              .groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().fillna(0))
    basket_encoded = basket.applymap(lambda x: 1 if x > 0 else 0)
    return basket_encoded

# Perform Apriori for each country
for country in top_countries:
    print(f"\n🔍 Analyzing Basket for Country: {country}")
    df_country = df_cleaned[df_cleaned['Country'] == country]
    basket = prepare_basket(df_country)
    # Show top 5 rows
    print(basket.head())  
    # Run Apriori
    frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)
    print(f"\n📈 Frequent Itemsets ({country}):")
    print(frequent_itemsets.head())

    # Generate Association Rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules = rules.sort_values(by='lift', ascending=False)

    # Convert frozensets to readable strings
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

    print(f"\n📌 Top Association Rules for {country}:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
