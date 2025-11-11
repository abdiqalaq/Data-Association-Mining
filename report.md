DSA 2040: Data Warehousing and Mining — Association Rule Mining

Author: Abdiqalaq Issack
Date: November 2025

1. Introduction
This report demonstrates the application of association rule mining on a real-world grocery transactions dataset. The goal is to discover relationships between items commonly purchased together, which can inform marketing strategies, cross-selling, and product placement.

2. Dataset Overview
Dataset Name: Groceries_dataset.csv
Source: Kaggle — Online Retail Groceries Dataset
Columns:
Member_number — unique customer ID
Date — transaction date
itemDescription — purchased item
Transactions: ~9835
Unique items: ~169

3. Data Preparation Process
Transaction ID creation:
Combined Member_number and Date to create a unique transaction ID.
Group items per transaction:
transactions = df.groupby('Transaction')['itemDescription'].apply(list).tolist()
One-hot encoding (basket format):
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
basket = pd.DataFrame(te_array, columns=te.columns_)
Handling missing data:
Removed duplicates and ensured all transactions were complete.

4. Frequent Itemset Mining
Algorithms: Apriori & FP-Growth
Support threshold: 0.01 (1%)
Top 10 Frequent Items (min_support=0.01):
Item	Support
whole milk	0.1579
other vegetables	0.1221
rolls/buns	0.1100
soda	0.0971
yogurt	0.0859
root vegetables	0.0696
tropical fruit	0.0678
bottled water	0.0607
sausage	0.0603
citrus fruit	0.0531

5. Apriori vs FP-Growth Performance Comparison
Metric	Apriori	FP-Growth
Runtime (seconds)	0.1082	0.1330
# Itemsets Found	69	69
Top 10 Items	Same	Same
Observations:
Both algorithms found the same frequent itemsets, confirming consistent results.
Runtime difference is minimal for this small dataset.
FP-Growth is more scalable for larger datasets, as it avoids generating candidate itemsets.

6. Association Rules Generation
Metric: Confidence ≥ 0.1
Analysis metrics: Support, Confidence, Lift
Top 3 Association Rules:
Antecedent	Consequent	Support	Confidence	Lift	Interpretation
yogurt	whole milk	0.0112	0.823	0.823	Yogurt buyers often buy whole milk — dairy combination
rolls/buns	whole milk	0.0140	0.127	0.804	Breakfast items bought together
other vegetables	whole milk	0.0148	0.122	0.769	Meal preparation association — vegetables and milk
Metric Interpretation:
Support: Frequency of itemsets in all transactions.
Confidence: Likelihood of consequent given antecedent.
Lift: Strength relative to chance; lift < 1 here due to the high frequency of whole milk.

7. Key Findings
Most frequent items: Whole milk, other vegetables, rolls/buns, yogurt.
Strong associations: Yogurt → Whole milk (high confidence).
Business insight:
Cross-promotions for dairy items.
Combo offers with bakery and vegetable items.
Algorithm insight:
Both Apriori and FP-Growth produce identical results.
FP-Growth is preferred for large-scale datasets due to better scalability.

8. Conclusion
Association rule mining identifies meaningful purchase patterns in grocery transactions.
Metrics such as support, confidence, and lift help assess the strength and usefulness of rules.
Insights can inform marketing strategies, cross-selling, and store layout planning.
Apriori is easy to implement but less scalable; FP-Growth is more efficient for larger datasets.