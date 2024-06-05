import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori
import streamlit as st

# Read data
coffe_shop = pd.read_csv('Coffe Shop Sales.csv')

# Pivot table
df_pivot = coffe_shop.pivot_table(index='transaction_number', columns='item', values='amount', aggfunc='sum').fillna(0)

# Ensure product names are unique by stripping unwanted whitespace
df_pivot.columns = df_pivot.columns.str.strip()

# Convert to integer
df_pivot = df_pivot.astype(int)

# Encode the pivot table
def encode(x):
    if x <= 0:
        return 0
    else:
        return 1

df_pivot = df_pivot.applymap(encode)

# Apply the apriori algorithm
support = 0.01
frequent_items = apriori(df_pivot, min_support=support, use_colnames=True)
frequent_items.sort_values('support', ascending=False, inplace=True)

# Generate the association rules
metric = 'lift'
min_threshold = 1
rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
rules.reset_index(drop=True, inplace=True)
rules.sort_values('confidence', ascending=False, inplace=True)

# Streamlit app
st.title("Association Rule Recommendation App")
st.markdown("This app helps you discover product recommendations based on customer purchase patterns.")

# Extract unique items from the rules
unique_items = list(set(item for sublist in rules["antecedents"].apply(list) for item in sublist))

# Display a selectbox with unique items
selected_item = st.selectbox("Select an item:", unique_items)

# Filter rules based on the selected item
filtered_rules = rules[rules["antecedents"].apply(lambda x: selected_item in [item for item in x])]

if filtered_rules.empty:
    st.warning(f"No recommendations found for {selected_item}.")
else:
    # Display top 3 recommendations with confidence and lift
    top_3_rules = filtered_rules.sort_values(by="confidence", ascending=False).head(3)
    st.subheader("Recommendation Results:")
    for index, row in top_3_rules.iterrows():
        consequent = list(row["consequents"])[0]
        confidence = round(row["confidence"], 3)
        lift = round(row["lift"], 3)
        st.write(
            f"- If the customer buys **{selected_item}**, "
            f"they also tend to buy **{consequent}** with {confidence * 100:.1f}% confidence "
            f"(lift: {lift})."
        )


