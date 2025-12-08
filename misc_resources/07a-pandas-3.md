# Working with Real Data in Pandas

So far we've worked with small, hand-crafted datasets to learn Pandas fundamentals. In practice, you'll load data from files, explore it to understand its structure and quality, manipulate it to suit your analysis needs, and save the results.

This notebook covers the essential skills for that workflow:

- Reading data from CSV files (and other formats)
- Exploring and diagnosing loaded data
- Setting and resetting index structures
- Filtering data with boolean indexing and queries
- Creating custom transformations with `apply()`
- Working with dates and times
- Saving processed data


```python
import numpy as np
import pandas as pd
```

## Loading Data from Files

The most common way to get data into Pandas is from CSV (comma-separated values) files. The `read_csv()` function handles this with many options to control how the data is loaded and interpreted.

### Basic CSV Reading

At its simplest, `read_csv()` takes a file path and returns a DataFrame.


```python
# For this example, we'll create a sample CSV file first
# In practice, you'll usually load existing files

sample_data = """date,product,quantity,price,region
2024-01-15,Widget A,100,25.50,North
2024-01-16,Widget B,150,30.00,South
2024-01-17,Widget A,200,25.50,East
2024-01-18,Widget C,75,45.00,West
2024-01-19,Widget B,125,30.00,North
2024-01-20,Widget A,,,South
2024-01-21,Widget C,90,45.00,East
"""

# Write to file
with open('data/07a_sales_sample.csv', 'w') as f:
    f.write(sample_data)

# Load it back
df = pd.read_csv('data/07a_sales_sample.csv')
df
```

Notice that Pandas automatically:
- Used the first row as column names
- Assigned a numeric index starting at 0
- Inferred data types for each column (minimal, strings and floats)
- Recognized empty values as `NaN`

### Initial Data Exploration

Before diving into analysis, we need to understand what we're working with. Pandas provides several tools for quick data exploration.


```python
# Get basic information about the DataFrame
df.info()
```

The `info()` method shows:
- Number of rows and columns
- Column names and data types
- Non-null counts (helps identify missing data)
- Memory usage


```python
# Check data types explicitly
df.dtypes
```


```python
# Get summary statistics for numeric columns
df.describe()
```

The `describe()` method provides count, mean, standard deviation, min, quartiles, and max for each numeric column. This is useful for spotting outliers and understanding data distributions.


```python
# Check unique values in a column
print("Unique products:")
print(df['product'].unique())

print("\nNumber of unique products:")
print(df['product'].nunique())
```


```python
# Count occurrences of each value
print("Product value counts:")
print(df['product'].value_counts())

print("\nRegion value counts:")
print(df['region'].value_counts())
```

The `value_counts()` method is particularly useful for understanding the distribution of categorical data. It returns a Series with counts sorted in descending order by default.

### Controlling Data Types on Load

Notice that the `date` column was loaded as an object (string) rather than an object type that represents dates. We can control how columns are interpreted using parameters to `read_csv()`.


```python
# Specify data types and parse dates
df = pd.read_csv('data/07a_sales_sample.csv',
                 dtype={'product': 'string', 'region': 'string'},
                 parse_dates=['date'])

df.info()
```

Now the `date` column is properly recognized as a `datetime64[ns]` type, which enables time-based operations we'll explore later.

**Note on data types:** When you access a single value from a datetime column, Pandas returns a `pd.Timestamp` object (not a Python `datetime`). This is Pandas' scalar datetime type - think of it as the single-value version of a `datetime64` array. It's fully compatible with Python's `datetime` but has additional Pandas-specific features.


```python
# Check the column dtype
print("Column dtype:", df['date'].dtype)

# Access a single date value
single_date = df.loc[0, 'date']
print("\nSingle value:", single_date)
print("Type:", type(single_date))

# Timestamp is compatible with datetime operations
print("\nYear:", single_date.year)
print("Month:", single_date.month)
```

### Handling Missing Values on Load

You can specify additional values that should be treated as missing (beyond the defaults like empty strings and 'NA').


```python
# Create a file with various missing value representations
messy_data = """name,score,status
Alice,95,active
Bob,N/A,inactive
Charlie,88,active
Diana,unknown,active
Eve,92,n/a
"""

with open('data/07a_messy_sample.csv', 'w') as f:
    f.write(messy_data)

# Load with custom missing value indicators
df_messy = pd.read_csv('data/07a_messy_sample.csv',
                       na_values=['N/A', 'unknown', 'n/a'])

print(df_messy)
print("\nNull value counts:")
print(df_messy.isna().sum())
```

### Setting the Index on Load

Often, one column naturally serves as an index (like dates or IDs). You can set this during loading.


```python
# Use date as the index
df_indexed = pd.read_csv('data/07a_sales_sample.csv',
                         parse_dates=['date'],
                         index_col='date')

df_indexed
```

This can be more efficient than loading and then setting the index separately, especially with large files.

### Memory Diagnostics for Larger Datasets

When working with larger datasets, memory usage becomes important. The `memory_usage()` method helps identify which columns consume the most memory.


```python
# Create a larger sample dataset
np.random.seed(42)
large_df = pd.DataFrame({
    'id': range(10000),
    'value': np.random.randn(10000),
    'category': np.random.choice(['A', 'B', 'C', 'D'], 10000),
    'date': pd.date_range('2020-01-01', periods=10000, freq='h'),
    'description': ['Item ' + str(i) for i in range(10000)]
})

# Check memory usage by column
print("Memory usage by column:")
print(large_df.memory_usage(deep=True))

print("\nTotal memory usage:")
print(f"{large_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

The `deep=True` parameter includes the actual memory used by object types like strings, which can be significant. Strings often represent categorical data, for which Pandas has a specific data type. By converting the `category` column to a categorical type we save memory *and* enable valuable functionality for that type (explored later).


```python
# Optimize by converting category column to categorical dtype
large_df['category'] = large_df['category'].astype('category')

print("Memory usage after optimization:")
print(large_df.memory_usage(deep=True))

print("\nTotal memory usage:")
print(f"{large_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

Converting columns with repeated values to the `category` dtype can dramatically reduce memory usage.

### Writing Data to CSV

After processing data, you'll often need to save the results. The `to_csv()` method handles this.


```python
# Save to CSV (default: includes index)
df.to_csv('data/07a_sales_processed.csv')

# Verify what was written
with open('data/07a_sales_processed.csv', 'r') as f:
    content = f.read()
    print(content[:200], '...')  # First 200 characters
```

Note: The leading comma in the header is expected - it's an empty column header for the index column.


```python
# Save without index (common for data without meaningful index)
df.to_csv('data/07a_sales_no_index.csv', index=False)

with open('data/07a_sales_no_index.csv', 'r') as f:
    print(f.read())
```


```python
# Save only specific columns
df.to_csv('data/07a_sales_subset.csv', 
          columns=['date', 'product', 'quantity'],
          index=False)

# Verify
pd.read_csv('data/07a_sales_subset.csv')
```

Pandas' `read_csv` has many other options...


```python
help(pd.read_csv)
```

### Other File Formats

Pandas supports many file formats beyond CSV. Excel files are particularly common in business settings.


```python
# Write to Excel (requires openpyxl or xlsxwriter package)
try:
    df.to_excel('data/07a_sales.xlsx', sheet_name='Sales', index=False)
    print("Excel file written successfully")
    
    # Read it back
    df_excel = pd.read_excel('data/07a_sales.xlsx', sheet_name='Sales')
    print("\nLoaded from Excel:")
    print(df_excel.head())
    
except ImportError:
    print("Excel support requires: pip install openpyxl")
```

For large datasets, CSV is generally faster and more reliable than Excel.

**Note:** Excel operations in Pandas require additional packages. Though Pandas relies on `openpyxl` for this, it is not identified as a default dependency, so is not added by conda during `conda install pandas`. Use `conda install openpyxl` in your `insy6500` environment if required. You do not need to import `openpyxl` - that is handled by Pandas.

## Index Operations

The index is more than just row numbers—it's a powerful tool for data organization and access. Understanding when and how to set an index is key to efficient Pandas use.

### Setting an Index

The `set_index()` method converts a column into the index.


```python
# Start with default numeric index
print("Original DataFrame:")
print(df.head())
print("\nIndex:", df.index)
```


```python
# Set date as index
df_dated = df.set_index('date')
print(df_dated)
print("\nIndex:", df_dated.index)
```

Remember that most Pandas methods return a new DataFrame rather than modifying the original. To update the original, either reassign it or use `inplace=True`.


```python
# Option 1: Reassign
df = df.set_index('date')

# Option 2: In-place (equivalent to above)
# df.set_index('date', inplace=True)
```

### When to Use an Index

Setting a meaningful index is beneficial when:

1. **Time series data**: Date/time indexes enable powerful time-based operations
2. **Unique identifiers**: Customer IDs, product codes, etc.
3. **Frequent lookups**: If you often access rows by a specific value
4. **Merging data**: Indexes are used in join operations

Keep the default numeric index when:
- No column naturally serves as an identifier
- You primarily work with all rows at once
- The data structure may change frequently

### Resetting the Index

The `reset_index()` method converts the index back into a regular column and creates a new default numeric index.


```python
# Current state: date is the index
print("Before reset:")
print(df.head())

# Reset to default numeric index
df_reset = df.reset_index()
print("\nAfter reset:")
print(df_reset.head())
```

### The `drop` Parameter

When resetting an index, you can choose whether to keep the old index as a column or discard it.


```python
# Reset and drop the old index
df_dropped = df.reset_index(drop=True)
print("After reset with drop=True:")
print(df_dropped.head())
```

This is useful when the index was created for temporary operations and isn't needed in the final result.

### Common Index Patterns

Here are some typical scenarios where index operations are useful.


```python
# Pattern 1: Fixing data structure after aggregation
# groupby() splits data into groups and applies aggregation functions
# Here we group by product and sum the quantities
# The grouping column (product) automatically becomes the index
grouped = df.groupby('product')['quantity'].sum()
print("Grouped result (product becomes index):")
print(grouped)

# Reset to make product a column again - useful for further analysis or output
grouped_df = grouped.reset_index()
print("\nAfter reset (product is now a column):")
print(grouped_df)
```


```python
# Pattern 2: Preparing for time series analysis
# Create sample time series data
ts_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=7, freq='D'),
    'sales': [100, 120, 115, 130, 125, 140, 135],
    'costs': [70, 85, 80, 90, 88, 95, 92]
})

print("Before setting date index:")
print(ts_data)

# Set date as index for time series operations
ts_data = ts_data.set_index('date')
print("\nAfter setting date index:")
print(ts_data)

# Now we can use date-based selection
print("\nSales for first 3 days:")
print(ts_data.loc['2024-01-01':'2024-01-03'])
```

**Note on date slicing:** The `.loc['2024-01-01':'2024-01-03']` syntax works here because `date` is the index *and* is a `datetime64` type. If `date` were just a regular column (not the index), you'd need to use boolean indexing instead: `ts_data[ts_data['date'].between('2024-01-01', '2024-01-03')]`. The datetime index enables this convenient slicing syntax.

## Boolean Indexing and Queries

We've seen basic boolean indexing before. Now we'll explore more complex filtering patterns that are essential for real data analysis.


```python
# Reset our working DataFrame for these examples
df = pd.read_csv('data/07a_sales_sample.csv', parse_dates=['date'])
print(df)
```

### Multiple Conditions with Boolean Indexing

Complex filters require combining conditions with `&` (and) or `|` (or). Each condition must be wrapped in parentheses.


```python
# Find high-value sales (quantity > 100 AND price > 30)
high_value = df[(df['quantity'] > 100) & (df['price'] > 30)]
print("High value sales:")
print(high_value)
```

The empty result is expected - our small sample dataset doesn't have any sales with *both* quantity > 100 *and* price > 30. Widget B has the right price (30.0) but we need strictly greater than 30. This illustrates how multiple conditions can filter data down to nothing if they're too restrictive. Let's try a more lenient version:


```python
# More useful: Find high-value sales (quantity > 100 AND price >= 30)
high_value = df[(df['quantity'] > 100) & (df['price'] >= 30)]
print("High value sales (quantity > 100 AND price >= 30):")
print(high_value)

# OR conditions: Find Widget A OR Widget C sales
widget_ac = df[(df['product'] == 'Widget A') | (df['product'] == 'Widget C')]
print("\nWidget A or C sales:")
print(widget_ac)
```


```python
# More complex: Widget A in North OR South regions with quantity > 100
complex_filter = df[
    (df['product'] == 'Widget A') & 
    (df['region'].isin(['North', 'South'])) &
    (df['quantity'] > 100)
]
print("Complex filter result:")
print(complex_filter)
```

Again, an empty result - a good lesson about **over-specification**. We're asking for Widget A, in North or South regions, with quantity > 100. Our Widget A sales are:
- 100 units in North (fails the quantity test)
- 200 units in East (fails the region test)
- NaN in South (fails the quantity test)

When debugging empty filter results, relax constraints one at a time to see which condition is eliminating your data.

**Important:** Use `&` and `|` for element-wise boolean operations in Pandas, not `and` and `or` (which are for Python boolean values only).

### The `isin()` Method

As demonstrated above, when checking if values are in a list of options, `isin()` is cleaner than multiple `|` conditions.


```python
# Instead of: (df['region'] == 'North') | (df['region'] == 'South') | (df['region'] == 'East')
# Use isin:
selected_regions = df[df['region'].isin(['North', 'South', 'East'])]
print("Sales in North, South, or East:")
print(selected_regions)
```


```python
# Can also use negation with ~
not_west = df[~df['region'].isin(['West'])]
print("\nSales NOT in West:")
print(not_west)
```

### The `query()` Method

For complex filters, `query()` provides a more readable alternative to boolean indexing. It accepts a string expression and evaluates it against the DataFrame.

The string is interpreted as a Python expression where column names become variables. So instead of writing `df[(df['quantity'] > 100)]`, you can write `df.query('quantity > 100')`. This is especially helpful for complex multi-condition filters that would otherwise require many parentheses and ampersands.


```python
# Boolean indexing version
result1 = df[(df['quantity'] > 100) & (df['price'] < 30)]

# query() version - more readable!
result2 = df.query('quantity > 100 and price < 30')

print("Using query():")
print(result2)
```


```python
# query() with string matching
widget_b_sales = df.query('product == "Widget B"')
print("Widget B sales:")
print(widget_b_sales)
```

**Using variables in queries:** The `@` symbol tells `query()` to look for a variable in the surrounding Python scope rather than treating it as a column name. Without `@`, `query('quantity >= min_quantity')` would look for a column called `min_quantity` in the DataFrame (and fail). With `@min_quantity`, it uses the Python variable's value.


```python
# query() can reference variables with @
min_quantity = 100
target_region = 'North'

result = df.query('quantity >= @min_quantity and region == @target_region')
print(f"Sales >= {min_quantity} in {target_region}:")
print(result)
```

### Combining `query()` with `loc[]` for Filtered Assignment

You can combine filtering with assignment to modify specific subsets of data.


```python
# Create a copy to demonstrate
df_modified = df.copy()

# Add a discount column
df_modified['discount'] = 0.0

# Apply 10% discount to high-quantity orders using boolean indexing
high_qty_mask = df_modified['quantity'] > 100
df_modified.loc[high_qty_mask, 'discount'] = 0.10

print("After applying discounts:")
print(df_modified)
```


```python
# More complex: apply different discounts based on multiple conditions
df_modified = df.copy()
df_modified['discount'] = 0.0

# 15% discount for large orders of Widget A
mask1 = (df_modified['product'] == 'Widget A') & (df_modified['quantity'] > 150)
df_modified.loc[mask1, 'discount'] = 0.15

# 10% discount for all other large orders
mask2 = (df_modified['quantity'] > 100) & (df_modified['discount'] == 0)
df_modified.loc[mask2, 'discount'] = 0.10

print("After tiered discounts:")
print(df_modified[['product', 'quantity', 'price', 'discount']])
```

**Note:** When making filtered assignments, always use `.loc[]` to avoid the "SettingWithCopyWarning" that can indicate subtle bugs. This aligns with previous guidance for when to use various subsetting methods in Pandas.

## Custom Transformations with `apply()`

While vectorized operations (like `df['price'] * 1.1`) are fast and should be used when possible, sometimes you need custom logic that doesn't fit a simple formula. That's where `apply()` comes in.

### Basic `apply()` with Lambda Functions

The `apply()` method takes a function and applies it to each element (or row/column) of a Series or DataFrame.


```python
# Simple transformation: categorize quantities
df['quantity_category'] = df['quantity'].apply(
    lambda x: 'High' if x > 150 else 'Medium' if x > 100 else 'Low'
)

print(df[['quantity', 'quantity_category']])
```


```python
# String manipulation: extract first word from product name
df['product_type'] = df['product'].apply(lambda x: x.split()[0])

print(df[['product', 'product_type']])
```

### When to Use `apply()`

Use `apply()` when:
- The operation requires conditional logic (if/else)
- You need to call functions that aren't vectorized
- The transformation depends on multiple conditions

Avoid `apply()` when:
- A vectorized operation exists (e.g., `df['A'] + df['B']`)
- A built-in Pandas method handles your case

Vectorized operations are much faster than `apply()` for large datasets.


```python
# Example: Vectorized approach is better when possible

# SLOW (using apply)
# df['total'] = df.apply(lambda row: row['quantity'] * row['price'], axis=1)

# FAST (vectorized)
df['total'] = df['quantity'] * df['price']

print(df[['quantity', 'price', 'total']])
```

### Applying Functions to Rows

By default, `apply()` works on columns. Use `axis=1` to work with rows.


```python
# Function that uses multiple columns
def calculate_status(row):
    if pd.isna(row['quantity']) or pd.isna(row['price']):
        return 'Incomplete'
    elif row['quantity'] * row['price'] > 5000:
        return 'High Value'
    elif row['quantity'] > 150:
        return 'High Volume'
    else:
        return 'Standard'

df['order_status'] = df.apply(calculate_status, axis=1)

print(df[['quantity', 'price', 'order_status']])
```

**Note:** Applying functions row-wise (`axis=1`) is generally slower than column-wise operations. When performance matters with large datasets, consider vectorized alternatives or NumPy functions.

## Working with Dates and Times

Time series data is ubiquitous in data analysis. Pandas provides excellent support for dates and times, which we'll explore in depth in a future lesson (12a). For now, we'll cover the basics you need for most analyses.

### Converting to Datetime

We've seen the `parse_dates` parameter in `read_csv()`. You can also convert existing columns with `pd.to_datetime()`.


```python
# Create DataFrame with date strings
date_data = pd.DataFrame({
    'date_str': ['2024-01-15', '2024-02-20', '2024-03-10'],
    'value': [100, 200, 150]
})

print("Before conversion:")
print(date_data.dtypes)

# Convert to datetime
date_data['date'] = pd.to_datetime(date_data['date_str'])

print("\nAfter conversion:")
print(date_data.dtypes)
```


```python
# pd.to_datetime() can handle various formats with format='mixed'
messy_dates = pd.Series([
    '2024-01-15',
    'January 15, 2024',
    '01/15/2024',
    '15-Jan-2024'
])

# Use format='mixed' to handle multiple date formats
clean_dates = pd.to_datetime(messy_dates, format='mixed')
print("Cleaned dates:")
print(clean_dates)
```

### The `dt` Accessor

Once a column is datetime type, the `.dt` accessor provides many useful properties and methods.


```python
# Using our sales data with dates
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_name'] = df['date'].dt.day_name()

print(df[['date', 'year', 'month', 'day', 'day_name']])
```


```python
# Useful for grouping by time periods
by_day = df.groupby(df['date'].dt.day_name())['quantity'].mean()
print("Average quantity by day of week:")
print(by_day)
```

### Common Datetime Operations


```python
# Filter by date range
jan_2024 = df[df['date'].dt.month == 1]
print("January 2024 sales:")
print(jan_2024[['date', 'product', 'quantity']])
```


```python
# Find weekend sales (Saturday=5, Sunday=6)
df['is_weekend'] = df['date'].dt.dayofweek >= 5
weekend_sales = df[df['is_weekend']]

print("Weekend sales:")
print(weekend_sales[['date', 'day_name', 'quantity']])
```


```python
# Calculate time differences
df_sorted = df.sort_values('date')
df_sorted['days_since_prev'] = df_sorted['date'].diff().dt.days

print("Days between sales:")
print(df_sorted[['date', 'days_since_prev']])
```

**Preview of 12a:** We'll cover time series in much more detail later, including resampling (aggregating by week/month), rolling windows (moving averages), time zones, and more advanced datetime operations. For now, these basics will handle most common needs.

## Putting It All Together: A Complete Workflow

Let's combine everything we've learned in a realistic data processing workflow:

1. Load
2. Explore
3. Clean
4. Feature Engineering
5. Filter and Analyze
6. Save



```python
# Create a more realistic sample dataset
extended_data = """date,product,quantity,price,region,customer_type
2024-01-15,Widget A,100,25.50,North,retail
2024-01-16,Widget B,150,30.00,South,wholesale
2024-01-17,Widget A,200,25.50,East,retail
2024-01-18,Widget C,75,45.00,West,retail
2024-01-19,Widget B,125,30.00,North,wholesale
2024-01-20,Widget A,,,South,retail
2024-01-21,Widget C,90,45.00,East,wholesale
2024-01-22,Widget A,175,25.50,North,retail
2024-01-23,Widget B,200,30.00,West,wholesale
2024-01-24,Widget C,50,45.00,South,retail
"""

with open('data/07a_extended_sales.csv', 'w') as f:
    f.write(extended_data)
```


```python
# Step 1: Load data with appropriate types
sales = pd.read_csv('data/07a_extended_sales.csv',
                    parse_dates=['date'],
                    dtype={'region': 'category', 'customer_type': 'category'})

print("Step 1 - Loaded data:")
print(sales.info())
```


```python
# Step 2: Explore the data
print("\nStep 2 - Data exploration:")
print("\nFirst few rows:")
print(sales.head())

print("\nSummary statistics:")
print(sales.describe())

print("\nMissing values:")
print(sales.isna().sum())

print("\nValue counts by region:")
print(sales['region'].value_counts())
```


```python
# Step 3: Clean the data
print("\nStep 3 - Cleaning:")

# Drop rows with missing critical data
# The 'subset' parameter specifies which columns to check for NaN
# A row is dropped only if it has NaN in ANY of these specified columns
sales_clean = sales.dropna(subset=['quantity', 'price'])
print(f"Removed {len(sales) - len(sales_clean)} rows with missing data")

# Set date as index for time-based operations
sales_clean = sales_clean.set_index('date')
```


```python
# Step 4: Create derived columns
print("\nStep 4 - Feature engineering:")

# Calculate revenue
sales_clean['revenue'] = sales_clean['quantity'] * sales_clean['price']

# Add day of week
sales_clean['day_of_week'] = sales_clean.index.day_name()

# Categorize order size
sales_clean['order_size'] = sales_clean['quantity'].apply(
    lambda x: 'Large' if x >= 175 else 'Medium' if x >= 100 else 'Small'
)

print(sales_clean[['quantity', 'price', 'revenue', 'order_size']].head())
```


```python
# Step 5: Filter and analyze
print("\nStep 5 - Analysis:")

# High-value wholesale orders
high_value_wholesale = sales_clean.query(
    'customer_type == "wholesale" and revenue > 5000'
)
print("\nHigh-value wholesale orders:")
print(high_value_wholesale[['product', 'region', 'revenue']])

# Average revenue by customer type and region
# observed=True prevents warning about future behavior change with categorical data
avg_revenue = sales_clean.groupby(['customer_type', 'region'], observed=True)['revenue'].mean()
print("\nAverage revenue by customer type and region:")
print(avg_revenue.round(2))
```


```python
# Step 6: Save processed data
print("\nStep 6 - Saving results:")

# Save full processed dataset
sales_clean.to_csv('data/07a_sales_processed_final.csv')
print("Saved processed data to 07a_sales_processed_final.csv")

# Save summary report
summary = sales_clean.groupby(['product', 'customer_type'], observed=True).agg({
    'quantity': 'sum',
    'revenue': 'sum',
    'region': 'count'
}).rename(columns={'region': 'num_orders'})

summary.to_csv('data/07a_sales_summary.csv')
print("Saved summary report to 07a_sales_summary.csv")
print("\nSummary:")
print(summary)
```

## Summary

In this notebook, we've covered the essential skills for working with real data in Pandas:

**File I/O:**
- Loading CSV files with `read_csv()` and controlling data types, date parsing, and missing values
- Using `info()`, `describe()`, `value_counts()`, and `memory_usage()` to understand loaded data
- Saving data with `to_csv()` and basic Excel operations

**Index Operations:**
- Setting meaningful indexes with `set_index()`
- Resetting to default numeric indexes with `reset_index()`
- Understanding when to use a custom index vs. default numeric index

**Boolean Indexing:**
- Filtering with complex conditions using `&`, `|`, and `~`
- Using `isin()` for cleaner membership tests
- Writing readable filters with `query()`
- Combining filters with `loc[]` for conditional assignment

**Custom Transformations:**
- Using `apply()` with lambda functions for custom logic
- Understanding when vectorized operations are better than `apply()`
- Applying functions row-wise vs. column-wise

**DateTime Basics:**
- Converting strings to datetime with `pd.to_datetime()`
- Extracting date components with the `.dt` accessor
- Filtering and grouping by date attributes

These skills form the foundation for most data analysis workflows. You can now load real datasets, explore their structure, clean and transform them, and save the results—the core loop of practical data work.
