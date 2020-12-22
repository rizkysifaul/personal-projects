import pandas as pd
import numpy as np
import streamlit as st
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from collections import Counter

st.title("Market Basket Analysis Application")
st.subheader("Using Apriori Algorithm")
st.write("Welcome to the application, main goal of this app is to know what rules can be when we bought an item(s).")
st.write("Let's check the transaction data.")

data = pd.read_csv('store_data.csv',header=None, sep=';',names=['Transaction'])
st.write(data)

st.write('There are 22 transaction, and product they bought from Wine, Apple and many more.')
st.write("For analyzing this data, we need to transforming this data to better angle.")



data = pd.read_csv('store_data.csv',header=None,names=['Trans','1','2','3','4','5'])
a=[]
for i in range(21):
    aa = data.iloc[i].values
    for x in aa:
        if x not in a:
            a.append(x)
        else:
            continue
data_count = pd.DataFrame()

for number in range(22):
    count = []
    for i in a:
        if i in list(dict(Counter(data.iloc[number].values)).keys()):
            count.append(dict(Counter(data.iloc[number].values))[i])
        else:
            count.append(0)

    data_count = data_count.append([count])
data_count = data_count.reset_index()
data_count.columns = ['id']+a
data_count = data_count.drop(['id',np.nan],axis=1)

st.write(data_count)

st.subheader("Selecting Support, Lift and Confidence")
st.write("After transforming our data, let's select our parameter to know what rules should we used.")
param_sup = st.slider('Select your minimum support: (Note: More higher, less rules and vice versa)', 0.0, 1.0, 0.5)
param_conf = st.slider('Select your minimum confidence: (Note: Note: More higher, less rules and vice versa)', 0.0, 1.0, 0.5)
param_lift = st.slider('Select your minimum lift: (Note: More than 1 means more likely can be used, 1 means no correlation, less than 1 is less likely)', 0.5, 2.0, 1.0)

if st.button("Checking your likely rule!"):
    frequent_itemsets = apriori(data_count, min_support=param_sup, use_colnames=True)
    frequent_itemsets = frequent_itemsets.sort_values('support',ascending=False)
    st.write("Frequent Itemset")
    st.write(frequent_itemsets)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=param_lift)
    rules = rules[rules['confidence'] >= param_conf]
    rules = rules[['antecedents', 'consequents', 'support','confidence','lift']]
    st.write("The Rule")
    st.write(rules)