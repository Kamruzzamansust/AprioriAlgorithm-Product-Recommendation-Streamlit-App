import numpy as np
import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import association_rules, apriori
import streamlit as st


coffe_shop = pd.read_csv(r'Coffe Shop Sales.csv')

df_pivot = coffe_shop.pivot_table(index='transaction_number',columns ='item',values = 'amount',aggfunc='sum').fillna(0)

df_pivot = df_pivot.astype(int)

def encode(x):
    if x <=0:
        return 0
    else:
        return 1
df_pivot = df_pivot.applymap(encode)

support = 0.01 
frequent_items = apriori(df_pivot, min_support=support, use_colnames=True)
frequent_items.sort_values('support', ascending=False)
metric = 'lift'
min_treshold = 1

rules = association_rules(frequent_items, metric=metric, min_threshold=min_treshold)[['antecedents','consequents','support','confidence','lift']]
rules.reset_index(drop=True).sort_values('confidence',ascending=False, inplace = True)
print(rules)

st.title("Association Rule Recommendation App")
st.markdown(
    "This app helps you discover product recommendations based on customer purchase patterns."
)


selected_item = st.selectbox("Select an item:", rules["antecedents"].apply(
    lambda x: list(x)[0]
))



filtered_rules = rules[rules["antecedents"].apply(
    lambda x: list(x)[0] == selected_item)]

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

