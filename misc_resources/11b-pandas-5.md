# GroupBy Advanced & Merge/Join

## Introduction

This lecture extends our groupby knowledge and introduces data combination techniques:

1. GroupBy Advanced Patterns: `transform()` and `apply()` for complex operations
2. Merge and Join: Combining DataFrames based on common keys

These are essential patterns for real data analysis where you need to:

- Add group statistics back to original rows (transform)
- Perform custom group operations (apply)
- Combine data from multiple sources (merge/join)


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load diamonds dataset
diamonds = sns.load_dataset('diamonds')
df = diamonds.sample(n=5000, random_state=42)
```

## User-Defined Functions with `agg`, `groupby`, etc.

Last lecture we learned aggregation: each group is represented as a single row with one or more statistics.


```python
# Aggregation: many rows → one row per group
df.groupby('cut', observed=False)['price'].mean()
```

Before we dive deeper, let's clarify something important: you can pass *any function* to `groupby operations` - built-in, custom named functions, or lambdas.

We've seen built-in functions:


```python
df.groupby('cut', observed=False)['price'].mean()
df.groupby('cut', observed=False)['price'].agg('mean')
```

But you can also write your own functions and pass them the same way.


```python
# Define a custom aggregation function
def price_range(prices):
    """Calculate the range (max - min) of prices"""
    return prices.max() - prices.min()

# Use it with agg
range_by_cut = df.groupby('cut', observed=False)['price'].agg(price_range)
print("Price range by cut:")
print(range_by_cut)
```

Your function will receive a Series (each group's data, one per iteration). For `agg`, it must return a single value. Other operations will have different expectations of the return value.

Lambdas are just anonymous functions - useful for one-liners but identical in concept to user defined functions.

The following code blocks are equivalent implementations of "de-meaning" the group.


```python
# Named function
def subtract_mean(values):
    return values - values.mean()

result1 = df.groupby('cut', observed=False)['price'].transform(subtract_mean)
```


```python
# Lambda (anonymous function)
result2 = df.groupby('cut', observed=False)['price'].transform(lambda x: x - x.mean())

# Verify they're the same
print(f"Results are identical: {result1.equals(result2)}")
```

When to use each:

- Built-in functions (`mean`, `sum`) wherever possible
- Named functions for complex logic, reusable across multiple analyses, easier to test
- Lambdas for simple one-liners that aren't reused elsewhere


## GroupBy Advanced Patterns

### transform(): Keep the Original Shape

With `agg`, each group is reduced to a single line of output.

Sometimes you want group statistics but keep all original rows. That's what transform() does.


```python
# Add the average price for each cut as a new column
df['avg_price_for_cut'] = df.groupby('cut', observed=False)['price'].transform('mean')

# Check the result
print(df[['cut', 'price', 'avg_price_for_cut']].head(10))
print(df.shape)
```

Notice:

- Every row keeps its original position
- Each row gets the average price for its cut group
- Same shape in, same shape out

### Practical Use Case: Comparing to Group Average

Transform is perfect for "how does this compare to others in its group?"


```python
# Calculate how much each diamond deviates from its cut's average price
df['price_vs_cut_avg'] = df['price'] - df.groupby('cut', observed=False)['price'].transform('mean')

# Show diamonds that are much more expensive than their cut's average
expensive_for_cut = df[df['price_vs_cut_avg'] > 5000]
print(f"Diamonds priced $5000+ above their cut's average: {len(expensive_for_cut)}")
print("\nExamples:")
print(expensive_for_cut[['cut', 'price', 'avg_price_for_cut', 'price_vs_cut_avg']].head())
```

Like `agg`, the `transform` method passes each subset of data (the rows in each group) to the chosen function, `mean`, one at a time. This is done as a loop (e.g., `for group in groups`), but the transform operation is vectorized within each group.

### Transform with Custom Functions

As expected, you can use transform with user defined or lambda functions for custom calculations. The input is the group data, as usual, but in this case you must return all the rows, not a single statistic.

The following example demonstrates this, returning a series containing the z-scores within each cut group.

Below we apply this technique to add a column to the dataset and use that to find observations where the price is a within-group outlier.


```python
# Calculate z-scores within each cut group
df['price_zscore_by_cut'] = (df
    .groupby('cut', observed=False)['price']
    .transform(lambda x: (x - x.mean()) / x.std())
)

# Find extreme outliers (|z| > 2)
outliers = df[df['price_zscore_by_cut'].abs() > 2]
print(f"Price outliers within their cut group: {len(outliers)}")
print("\nExamples:")
print(outliers[['cut', 'price', 'price_zscore_by_cut']].head())
```

### When to Use transform()

Use transform() when you need to:
- Add group statistics back to every row
- Calculate within-group z-scores or percentiles
- Compare individual values to their group
- Keep the original DataFrame structure

Don't use transform() when:
- You just need summary statistics (use aggregation)
- You're creating a report (use aggregation)

### apply(): Custom Group Operations

When built-in aggregations aren't enough, use `apply()` with a custom function. Your function can return a scalar (single value per group), Series (multiple values per group), or DataFrame (multiple rows per group).

In the example below `top_3_expensive` takes a group and returns the rows with the three highest prices.


```python
# Get the top 3 most expensive diamonds for each cut
def top_3_expensive(group):
    return group.nlargest(3, 'price')

top_diamonds = df.groupby('cut', observed=False).apply(top_3_expensive, include_groups=False)
print(top_diamonds[['carat', 'price']])
```

`apply()` returns a new DataFrame with a hierarchical index - the group, then the original index.

Note that `include_groups=False` is specified in `apply`. This is required to avoid more FutureWarnings about deprecated behavior and makes it explicit that *do not* want to include the group column (i.e., `cut`) in the data sent to `apply`. In the future this will be the default behavior.

### apply() with Aggregation

You can return summary statistics that don't fit standard aggregation patterns.


```python
# Calculate the price range (max - min) and coefficient of variation for each cut
def price_stats(group):
    return pd.Series({
        'price_range': group['price'].max() - group['price'].min(),
        'cv': group['price'].std() / group['price'].mean(),
        'count': len(group)
    })

custom_stats = df.groupby('cut', observed=False).apply(price_stats, include_groups=False)
print(custom_stats)
```

### Performance Warning: apply() is Slow

apply() is flexible but slower than vectorized operations.


```python
# Slow: using apply for something simple
%timeit df.groupby('cut', observed=False).apply(lambda x: x['price'].mean(), include_groups=False)

# Fast: using built-in aggregation
%timeit df.groupby('cut', observed=False)['price'].mean()
```

Rule: Use built-in aggregations when possible. Only use apply() when you truly need custom logic.

### Summary: Choosing the Right GroupBy Method

Comparison of groupby operations:

| Method | Output Shape | Use Case | Performance | When to Use |
|--------|--------------|----------|-------------|-------------|
| `.agg()` | One row per group | Summary statistics |  Fast | Need summary report or statistics |
| `.transform()` | Same as input | Add group stats to each row |  Slower | Need to compare individuals to their group |
| `.apply()` | Varies | Custom operations | Slowest | Nothing else works; complex logic |

Decision Guide:

- Want one number per group? → Use `.agg()`
- Want to keep all rows? → Use `.transform()`
- Need complex custom logic? → Use `.apply()` (but check for vectorized alternatives first!)


## Merge and Join

### Understanding Relational Data: Keys and Relationships

Before learning the pandas syntax, let's understand WHY data is often split across multiple tables and HOW those tables relate to each other. This conceptual foundation will make merge and join operations intuitive rather than mysterious.

### The Problem: Data Lives in Multiple Places

Real-world data is rarely in a single table. Consider a _very simple_ online store:

- Customers table: `customer_id`, `name`, `email`, `city`
- Orders table: `order_id`, `customer_id`, `order_date`, `amount`
- Products table: `product_id`, `name`, `price`, `category`

Why separate tables instead of one giant table?

- Avoid duplication (store customer info once, not on every order)
- Different update frequencies (products change prices, orders don't)
- Different data sources (CRM system, transaction database, inventory system)

The challenge: How do we connect related data across tables?

### Primary and Foreign Keys

**Primary Key**: A column (or combination of columns) that uniquely identifies each row in a table.


```python
# Customers table - customer_id is the primary key
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],  # ← PRIMARY KEY (unique)
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix']
}).set_index('customer_id')
print("Customers (customer_id is PRIMARY KEY):")
print(customers)
```

**Foreign Key**: A column in one table that references the primary key in another table. It creates the link between tables.


```python
# Orders table - customer_id is a foreign key referencing customers
orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105, 106],
    'customer_id': [1, 2, 2, 3, 6, 1],  # ← FOREIGN KEY (references customers)
    'amount': [250, 180, 420, 310, 150, 290]
}).set_index('order_id')
print("\nOrders (customer_id is FOREIGN KEY):")
print(orders)
```

Notice: 

- Customer 2 (Bob) appears twice in orders - he placed multiple orders
- Customer 6 appears in orders but doesn't exist in customers (data quality issue!)
- Customers 4 and 5 have no orders

This is where merge/join operations help us connect the tables.

### Relationship Types

Understanding the relationship between tables helps you predict what merge operations will produce.

#### One-to-One (1:1)

Each row in Table A matches exactly one row in Table B. For example, Customers and Loyalty Profiles - each customer has one loyalty profile and each loyalty profile belongs to one customer.

1:1 relationships are unusual in practice - you would often just add these columns to the original table.

Use separate tables when the data comes from different systems or has different update patterns.

#### One-to-Many (1:M) or Many-to-One (M:1)

Each row in Table A can match multiple rows in Table B.

This is the *most common* relationship type in real data.

Example: Customers and orders (1:M, one customer can place many orders)


```python
print("Customers (the 'one' side):")
print(customers[['name']].head(3))
print("\nOrders (the 'many' side):")
print(orders[['customer_id', 'amount']].head(6))
```

The relationship between customers and orders is represented by copying the primary key from the "one side" as a foreign key on the "many side." Hence, orders includes `customer_id` as a reference (foreign key) to the customer table.

Customer 1 (Alice) has orders: 101, 106

Customer 2 (Bob) has orders: 102, 103

Customer 3 (Charlie) has orders: 104

#### Many-to-Many (M:N)

Each row in Table A can match multiple rows in Table B, and vice versa.

Example: Students and Courses (M:N, one student takes many courses, one course has many students)

Problem: You can't directly merge students and courses - there's no common key that makes sense!

Solution: Create a *junction table* (also called association table or bridge table) that connects them.


```python
# Cannot directly merge students and courses - no common key!
# Solution: junction table

students = pd.DataFrame({
    'student_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

courses = pd.DataFrame({
    'course_id': [101, 102, 103],
    'course_name': ['Math', 'Physics', 'Chemistry']
})

# Junction table connects them
enrollments = pd.DataFrame({
    'student_id': [1, 1, 2, 2, 3, 3],
    'course_id': [101, 102, 101, 103, 102, 103],
    'grade': ['A', 'B', 'A', 'C', 'B', 'A']
})
print(enrollments)
```

Now we can connect students → enrollments → courses to get complete information. Pandas implements this via the `merge` and `join` methods, which we'll cover shortly.

Note that the junction table can also include information about the association, in this case the student-course grade, which is only relevant in the context of that relationship.

### Relationship Types Summary

| Type | Example | Key Pattern | Merge Behavior |
|------|---------|-------------|----------------|
| 1:1 | Customer → Loyalty Profile | One FK value per row on both sides | Simple, rarely used |
| 1:M | Customer → Orders | One PK matches many FK values | Most common, inner/left/right matter |
| M:N | Students ↔ Courses | Requires junction table | Chain multiple merges |

### Why This Matters for Pandas Merge

Understanding relationships helps you:

1. Choose the right approach to designing and joining tables
2. Predict the output size:
   - 1:1 merge: Same number of rows as input (if all match)
   - 1:M merge: Number of rows = number on the "many" side
   - M:N: Requires junction table, can explode rows
3. Spot data quality issues:
   - Foreign key values with no matching primary key (orphaned records)
   - Unexpected duplicates in what should be 1:1

Now that you understand HOW tables relate, pandas merge operations will make intuitive sense.

### Creating Example Datasets

Let's create simple datasets to demonstrate merge operations clearly.


```python
# Customer data
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix']
})

# Order data
orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105, 106],
    'customer_id': [1, 2, 2, 3, 6, 1],
    'amount': [250, 180, 420, 310, 150, 290]
})

print("Customers:")
print(customers)
print("\nOrders:")
print(orders)
```

Notice:

- Customer 4 (Diana) and 5 (Eve) have no orders
- Customer 6 in orders doesn't exist in customers table
- Customer 1 (Alice) has multiple orders

### Inner Join: Only Matching Rows

Inner join keeps only rows where the key exists in BOTH DataFrames.


```python
# Inner join (default)
inner = pd.merge(customers, orders, on='customer_id', how='inner')
print(inner)
```

Result:
- 5 orders (customer 6's order excluded because no matching customer)
- Only customers 1, 2, 3 appear (they have orders)
- Customers 4, 5 excluded (no orders)

### Left Join: Keep All from Left

Left join keeps ALL rows from the left DataFrame, adding matching data from right.

The resulting table will have `NaN` values wherever there is no match.


```python
# Left join
left = pd.merge(customers, orders, on='customer_id', how='left')
print(left)
```

Result:
- All 5 customers appear
- Customers without orders have NaN in order columns
- Customer 6's order excluded (not in customers table)

### Right Join: Keep All from Right

Right join keeps ALL rows from the right DataFrame.

Like left join, unmatched data will generate `NaN` values.


```python
# Right join
right = pd.merge(customers, orders, on='customer_id', how='right')
print(right)
```

Result:
- All 6 orders appear
- Customer 6's order included, but customer details are NaN
- Customers 4, 5 excluded (no orders)

### Outer Join: Keep All from Both

Outer join keeps ALL rows from BOTH DataFrames.


```python
# Outer join
outer = pd.merge(customers, orders, on='customer_id', how='outer')
print(outer)
```

Result:
- All customers (1-5) appear
- All orders (including customer 6) appear
- NaN fills gaps where data doesn't match

### Join Behavior Summarized

- Inner:  A ∩ B (only matching); strict - both must agree
- Left:   All of A + matching from B; "Keep my data (left), add theirs if it exists"
- Right:  All of B + matching from A; rarely used (just swap left / right and use left join)
- Outer:  A ∪ B (everything); generous, used for debugging

### Merging on Different Column Names

Often the key column has different names in each DataFrame.


```python
# Create product data with different column name
products = pd.DataFrame({
    'product_id': [101, 102, 103],
    'product_name': ['Widget', 'Gadget', 'Doohickey'],
    'category': ['Tools', 'Electronics', 'Home']
})

sales = pd.DataFrame({
    'sale_id': [1, 2, 3, 4],
    'prod_id': [101, 102, 101, 103],
    'quantity': [5, 3, 2, 7]
})

print("Products:")
print(products)
print("\nSales:")
print(sales)
```

Here products' primary key is `product_id`, but it is referenced in sales as `prod_id`. You can specify this explicitly in the merge:


```python
# Merge on different column names
merged = pd.merge(sales, products, 
                  left_on='prod_id', 
                  right_on='product_id', 
                  how='left')
print(merged)
```

Note that both key columns are kept. You can drop the redundant one if needed.

### Merging on Multiple Columns

Sometimes the unique key is a combination of columns (composite key).


```python
# Sales data with store and date
sales_detail = pd.DataFrame({
    'store': ['A', 'A', 'B', 'B'],
    'date': ['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02'],
    'revenue': [1000, 1200, 800, 950]
})

# Inventory data
inventory = pd.DataFrame({
    'store': ['A', 'A', 'B'],
    'date': ['2024-01-01', '2024-01-02', '2024-01-01'],
    'stock': [50, 45, 30]
})

# Merge on both store AND date
combined = pd.merge(sales_detail, inventory, on=['store', 'date'], how='left')
print(combined)
```

Store B on 2024-01-02 has no inventory data, so stock is NaN.

### Merging on Index

When the join key is in the index, you can use index parameters instead.


```python
# Create DataFrames with meaningful indexes
prices = pd.DataFrame({
    'product': ['Widget', 'Gadget', 'Doohickey'],
    'price': [19.99, 49.99, 29.99]
}).set_index('product')

ratings = pd.DataFrame({
    'product': ['Widget', 'Gadget', 'Gizmo'],
    'rating': [4.5, 4.2, 4.8]
}).set_index('product')

print("Prices:")
print(prices)
print("\nRatings:")
print(ratings)
```


```python
# Merge on index
merged = pd.merge(prices, ratings, left_index=True, right_index=True, how='outer')
print(merged)
```

### Simpler Index Join: .join()

For index-based joins, DataFrame.join() is simpler.


```python
# Same result, cleaner syntax
joined = prices.join(ratings, how='outer')
print(joined)
```

`join()` defaults to left join and uses indexes automatically.

Best practice: when defining tables, set the index to the primary key, if one exists, or use the default row index if one does not. Then use the simplified `join` syntax.

### Handling Duplicate Column Names

When both DataFrames have columns with the same name (other than the join key), use suffixes.


```python
# Both have 'price' column
df1 = pd.DataFrame({
    'product_id': [1, 2, 3],
    'price': [10, 20, 30]
})

df2 = pd.DataFrame({
    'product_id': [1, 2, 3],
    'price': [12, 22, 32]
})

# Merge with suffixes
merged = pd.merge(df1, df2, on='product_id', suffixes=('_old', '_new'))
print(merged)
```

Without suffixes, you'd get 'price_x' and 'price_y' (less meaningful).


```python
merged = pd.merge(df1, df2, on='product_id')
print(merged)
```

## concat(): Stacking DataFrames

### When to Use concat() vs merge()

Use `.concat()` when:

- Combining DataFrames with identical structure (same columns)
- Stacking monthly/regional datasets
- Appending new data to existing data

Use `.merge()` when:

- Combining related but different data
- Matching on key columns
- Data comes from different sources

### `concat` to Stack DataFrames

Combine DataFrames by vertically stacking rows (most common) by specifying `axis=0`.


```python
# January sales
jan_sales = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    'revenue': [1000, 1200, 1100]
})

# February sales
feb_sales = pd.DataFrame({
    'date': ['2024-02-01', '2024-02-02', '2024-02-03'],
    'revenue': [1300, 1250, 1400]
})

# Stack vertically
all_sales = pd.concat([jan_sales, feb_sales], axis=0)
print(all_sales)
```

Notice the index is preserved from each DataFrame. Often you want to reset it.


```python
# Stack with reset index
all_sales_clean = pd.concat([jan_sales, feb_sales], axis=0, ignore_index=True)
print(all_sales_clean)
```

You can also stack DataFrames horizontally (side-by-side) with `axis=1`, but `merge` / `join` are typically preferred for that to ensure the rows are aligned by key values. `concat` assumes rows align by position (index), which can be problematic.

### Handling Mismatched Columns

When DataFrames don't have identical columns, concat fills with NaN.


```python
# Different columns
df_a = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

df_b = pd.DataFrame({
    'B': [7, 8, 9],
    'C': [10, 11, 12]
})

# Concat fills missing columns with NaN
stacked = pd.concat([df_a, df_b], axis=0, ignore_index=True)
print(stacked)
```

### Practical Example: Combining Regional Data

Real-world scenario: combining sales data from multiple regions.


```python
# Regional sales data
northeast = pd.DataFrame({
    'region': ['Northeast'] * 3,
    'product': ['Widget', 'Gadget', 'Doohickey'],
    'sales': [1000, 1500, 800]
})

southeast = pd.DataFrame({
    'region': ['Southeast'] * 3,
    'product': ['Widget', 'Gadget', 'Doohickey'],
    'sales': [1200, 1300, 900]
})

west = pd.DataFrame({
    'region': ['West'] * 3,
    'product': ['Widget', 'Gadget', 'Doohickey'],
    'sales': [1800, 2000, 1100]
})

# Combine all regions
national = pd.concat([northeast, southeast, west], axis=0, ignore_index=True)
print(national)
print(f"\nTotal rows: {len(national)}")
```

Now you can analyze by region, product, or both.


```python
# Total sales by product across all regions
national.groupby('product')['sales'].sum()
```

## Real-World Example: Combining Everything

Let's combine groupby, merge, and visualization.


```python
# Calculate average price by cut from our diamonds sample
cut_avg = df.groupby('cut', observed=False).agg(
    avg_price=('price', 'mean'),
    count=('price', 'count')
).reset_index()

print("Average price by cut:")
print(cut_avg)
```


```python
# Add cut quality information to each diamond
df_enriched = pd.merge(df, cut_avg, on='cut', how='left', suffixes=('', '_avg'))

# Now each row knows its cut's average price
df_enriched['premium_over_avg'] = df_enriched['price'] - df_enriched['avg_price']

print("\nDiamonds with biggest premium over their cut average:")
print(df_enriched.nlargest(5, 'premium_over_avg')[
    ['cut', 'carat', 'price', 'avg_price', 'premium_over_avg']
])
```


```python
# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Average price by cut
sns.barplot(data=cut_avg, x='cut', y='avg_price', ax=axes[0])
axes[0].set_title('Average Price by Cut Quality')
axes[0].set_ylabel('Average Price ($)')
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: Distribution of premium over average
sns.histplot(data=df_enriched, x='premium_over_avg', bins=50, ax=axes[1])
axes[1].set_title('Distribution: Price vs Cut Average')
axes[1].set_xlabel('Price - Cut Average ($)')
axes[1].axvline(x=0, color='red', linestyle='--', label='Average')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```

## Key Takeaways

GroupBy Advanced:
- transform() keeps original shape, adds group stats to each row
- apply() allows custom group operations but is slower
- Use built-in aggregations when possible

Merge and Join:
- Inner join: only matching rows
- Left join: all from left, matching from right
- Right join: all from right, matching from left
- Outer join: all from both
- Use left_on/right_on for different column names
- Use on=[col1, col2] for composite keys

concat():
- Vertical stacking (axis=0): combine rows
- Horizontal stacking (axis=1): combine columns
- Use ignore_index=True to reset index
- Best for identical or similar structures

Next lecture: Data quality, validation, and handling missing data.
