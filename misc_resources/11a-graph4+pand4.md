# Matplotlib API and Pandas GroupBy Foundations

This lecture (11a) is being delivered on 10/30. Lecture 11b will be delivered as a recording to catch up for Tuesday's class (INFORMS).

We are behind schedule. This week we'll wrap up the loose ends of Pandas / Graphing before moving on to data quality and analysis next week. I'll post an updated schedule before we next meet. Test 2 will be moved out in that plan.

## Introduction

This lecture has two parts:

1. Matplotlib API Essentials: Understanding the structure beneath pandas and seaborn plotting
2. GroupBy Operations: The split-apply-combine pattern for data aggregation

By the end, you'll understand how to customize plots when needed and how to aggregate data by groups - one of the most powerful patterns in data analysis.

## Part 1: Matplotlib API Essentials

You've been using matplotlib through pandas (`.plot`) and seaborn - now let's see what's underneath so you can customize when needed.

### Why This Matters

When you call `df.plot()` or `sns.scatterplot()`, you're using matplotlib under the hood. Understanding the basic structure lets you:

- Create custom multi-plot layouts
- Fine-tune individual plot elements
- Combine plots in ways that high-level APIs don't support
- Debug when things don't look right


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load our familiar dataset
diamonds = sns.load_dataset('diamonds')
df = diamonds.sample(n=5000, random_state=42)

# basic seaborn scatter
sns.scatterplot(data=df, x='carat', y='price', alpha=0.3)
```

Note the first line of output - this is the axes returned by Seaborn after generating the plot. We can assign that to a variable for further manipulation with matplotlib...


```python
# sns returns the matplotlib axes; capture it for further manipulation
ax = sns.scatterplot(data=df, x='carat', y='price', alpha=0.3)
ax.set_xlabel('Carat')
ax.set_ylabel('Price ($)')
ax.set_title('Price vs Carat')
ax.grid(True, alpha=0.3)

# must then show the plot
plt.show()
```

The same is true of Pandas' `plot` methods.


```python
# basic pandas scatter
df.plot.scatter(x='carat', y='price', alpha=0.3)
```


```python
# with matplotlib
ax = df.plot.scatter(x='carat', y='price', alpha=0.3)
ax.set_xlabel('Carat')
ax.set_ylabel('Price ($)')
ax.set_title('Price vs Carat')
ax.grid(True, alpha=0.3)
plt.show()
```

### The Figure and Axes Model

Matplotlib has a hierarchy:

- Figure: The overall window/page (can contain multiple plots)
- Axes: An individual plot within the figure (despite the name, it's the whole plot, not just the axis lines)

Use `plt.subplots()` to create both and assign them to variables. Use those variables to modify the contents.

In the simplest case we can just use seaborn to generate a plot in the specified axes. This is equivalent to the previous method.


```python
# create figure and axes handles
fig, ax = plt.subplots()

# generate plot in ax
sns.scatterplot(data=df, x='carat', y='price', ax=ax)

# Now 'ax' is an Axes object you can customize
ax.set_xlabel('Carat')
ax.set_ylabel('Price ($)')
ax.set_title('Price vs Carat')
ax.grid(True, alpha=0.3)

# show the result
plt.show()
```

So when DO you need to create `fig, ax` first?

1. When you need to control figure size


```python
fig, ax = plt.subplots(figsize=(8, 3))
sns.scatterplot(data=df, x='carat', y='price', ax=ax)
plt.show()
```

2. When creating multiple subplots (not facets - that's seaborn)


```python
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
sns.scatterplot(data=df, x='carat', y='price', ax=axes[0])
sns.histplot(data=df, x='price', ax=axes[1])
plt.show()
```

3. When combining multiple plot types on the same axes


```python
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='carat', y='price', ax=ax)
ax.axhline(y=5000, color='red', linestyle='--')  # Add reference line`
plt.show()
```

In each example, `plt.show()` explicitly renders the current plot. In Jupyter notebooks, plots display automatically by default, so `plt.show()` is optional but harmless. In Python scripts, `plt.show()` is required to open a display window. The returned axes object allows further customization regardless of environment.

### Key Customization Methods

Once you have an `ax` (Axes object), you can customize it:


```python
fig, ax = plt.subplots(figsize=(8, 5))

# Create the plot - here we use matplotlib directly
ax.scatter(df['carat'], df['price'], alpha=0.3, color='steelblue')

# Customize labels and title
ax.set_xlabel('Carat Weight', fontsize=12)
ax.set_ylabel('Price (USD)', fontsize=12)
ax.set_title('Diamond Price vs Carat Weight', fontsize=14, fontweight='bold')

# Control axis limits
ax.set_xlim(0, 3)
ax.set_ylim(0, 20000)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--')

# Add legend if needed
ax.legend(['Diamond prices'], loc='upper left')

plt.tight_layout()
plt.show()
```

### Multiple Subplots

The real power comes when you need multiple plots arranged in a grid.


```python
# Create a 2x2 grid of plots
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

# axes is now a 2D array of Axes objects
# Flatten it for easier iteration
axes = axes.flatten()

# Plot different cut qualities
cuts = ['Fair', 'Good', 'Very Good', 'Premium']
for i, cut in enumerate(cuts):
    data = df[df['cut'] == cut]

    # generate plot using matplotlib directly
    # axes[i].scatter(data['carat'], data['price'], alpha=0.3)

    # or use seaborn instead
    sns.scatterplot(data=data, x='carat', y='price', ax=axes[i], alpha=0.3)

    axes[i].set_title(f'{cut} Cut')
    axes[i].set_xlabel('Carat')
    axes[i].set_ylabel('Price ($)')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

```

### To Add Here...

I'll add notes about how to control the style of your plots, color palettes, scaling, and saving the output for highest quality in your work. This will be referenced in HW4.

### Matplotlib - When?

Use matplotlib when you need precise control over subplot layouts and contents, or you need to annotate specific points or regions. Otherwise, lean into higher level APIs (pandas, seaborn). Consider: are you doing quick visualization (`pd.plot`), need stats or higher quality results (seaborn), or full control (matplotlib).

The benefits of each will become more apparent with experience and when we discuss best practices in visualization later this semester.

### Matplotlib Online Resources

Our textbook, _Python Data Science Handbook_, has an entire chapter dedicated to matplotlib. If you want to learn more about the somewhat arcane inner-workings of matplotlib, I recommend the following resources.

Here are the best online resources for diving deeper into matplotlib:

- [Official documentation](https://matplotlib.org/stable/tutorials/index.html)
  - Start with "Quick start guide" and "Introduction to pyplot"
- [Matplotlib Cheatsheets](https://matplotlib.org/cheatsheets/)
  - Beginner, intermediate, and advanced cheatsheets
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
  - Browse by plot type, find examples with code
- [Real Python Matplotlib Guide](https://realpython.com/python-matplotlib-guide/)
  - Step-by-step tutorial format


## Part 2: GroupBy Operations

The most important pattern in data aggregation: **split-apply-combine**.

### The Split-Apply-Combine Pattern

**Concept**: 
1. **Split** your data into groups based on some criteria
2. **Apply** a function to each group independently  
3. **Combine** the results back together

This is how you answer questions like:
- What's the average price for each cut quality?
- How many diamonds are in each color grade?
- What's the total carat weight by clarity?


```python
# Simple example: average price by cut
df.groupby('cut', observed=False)['price'].mean()
```

### Basic Aggregations

The most common pattern: group by one column, aggregate another.


```python
# Average price by cut
print("Average price by cut:")
print(df.groupby('cut', observed=False)['price'].mean())
```


```python
# Total carats by cut
print("\nTotal carats by cut:")
print(df.groupby('cut', observed=False)['carat'].sum())
```


```python
# Count of diamonds by color
print("\nCount by color:")
print(df.groupby('color', observed=False)['price'].count())
```


```python
# Or more simply:
print(df.groupby('color', observed=False).size())
```

There is a subtle but important difference here. 

- `.count()` - Counts **non-null values** in the specified column
- `.size()` - Counts **total rows** in each group

They're equivalent _only_ when there are no missing values.

**Use `.count()` when:**
- You specifically want to count non-null values
- You want counts for multiple columns: `df.groupby('color').count()`

**Use `.size()` when:**
- You want total group sizes (most common case)
- Cleaner syntax for "how many in each group?"

### Multiple Aggregations

Often you want several statistics at once:


```python
# Multiple aggregations for price
df.groupby('cut', observed=False)['price'].agg(['mean', 'median', 'std', 'min', 'max'])
```


```python
# Aggregate different columns with different functions
res = df.groupby('cut', observed=False).agg({
    'price': ['mean', 'median'],
    'carat': ['mean', 'sum'],
    'depth': 'mean'
})

print(res)
```

### Grouping by Multiple Columns

You can group by multiple columns to get finer-grained aggregations:


```python
# Average price by cut AND color
result = df.groupby(['cut', 'color'], observed=False)['price'].mean()
result
```


```python
# Look at the structure
print(f"\nType: {type(result)}")
print(f"Index type: {type(result.index)}")
print(f"\nFirst few index values:")
print(result.index[:5])
```

### What GroupBy Returns

Understanding what you get back is important:


```python
# The groupby object itself (before aggregation)
grouped = df.groupby('cut', observed=False)
print(f"Type: {type(grouped)}")
print(f"Groups: {grouped.ngroups}")
```


```python
# After aggregation, you get a Series or DataFrame
result = df.groupby('cut', observed=False)['price'].mean()
print(f"\nType after aggregation: {type(result)}")
print(f"Index: {result.index}")
```

The grouping column became the index!


```python
result
```


```python
# Convert index back to column with reset_index()
result_df = result.reset_index()
print(result_df)
```

### Practical Pattern: Named Aggregations

For clarity in your results, use named aggregations:


```python
# Clear, readable output
summary = df.groupby('cut', observed=False).agg(
    avg_price=('price', 'mean'),
    total_sales=('price', 'sum'),
    num_diamonds=('price', 'count'),
    avg_carat=('carat', 'mean')
)

print(summary)
```


```python
# This is especially useful when you reset_index
summary_df = summary.reset_index()
print(summary_df)
```

### Filtering Groups

Sometimes you want to filter entire groups, not just rows.


```python
# Only keep groups with more than 500 diamonds
large_groups = df.groupby('cut', observed=False).filter(lambda x: len(x) > 500)
print(f"\nOriginal data: {len(df)} rows")
print(f"After filtering: {len(large_groups)} rows")
print(f"\nRemaining cuts:")
print(large_groups['cut'].value_counts())
```

Here, `filter` is acting on grouped data, so `len(x)` gives the number of rows in each group.

### Combining GroupBy with Visualization

GroupBy results are perfect for visualization:


```python
# Calculate statistics
avg_price = df.groupby('cut', observed=False)['price'].mean().reset_index()
avg_price
```


```python
# Plot
fig, ax = plt.subplots(figsize=(6, 4))

ax.bar(avg_price['cut'].astype(str), avg_price['price'], 
       color='steelblue', alpha=0.7)
ax.set_xlabel('Cut Quality')
ax.set_ylabel('Average Price ($)')
ax.set_title('Average Diamond Price by Cut Quality')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

### Common Aggregation Functions

Quick reference for common operations:


```python
# Numeric aggregations
numeric_aggs = df.groupby('cut', observed=False)['price'].agg([
    'count',    # Number of items
    'sum',      # Total
    'mean',     # Average
    'median',   # Middle value
    'std',      # Standard deviation
    'var',      # Variance
    'min',      # Minimum
    'max',      # Maximum
    'first',    # First value
    'last'      # Last value
])

print(numeric_aggs)
```

### Real-World Example: Diamond Price Analysis

Combine what we've learned.


```python
# Create comprehensive summary by cut
analysis = df.groupby('cut', observed=False).agg(
    count=('price', 'count'),
    avg_price=('price', 'mean'),
    median_price=('price', 'median'),
    price_std=('price', 'std'),
    avg_carat=('carat', 'mean'),
    total_value=('price', 'sum')
).round(2)

print(analysis)
```


```python
# Visualize key findings
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Reset index so 'cut' becomes a column (seaborn expects tidy data)
analysis_reset = analysis.reset_index()

# Plot 1: Count by cut
sns.barplot(data=analysis_reset, x='cut', y='count', 
            color='steelblue', alpha=0.7, ax=axes[0])
axes[0].set_title('Number of Diamonds by Cut')
axes[0].set_xlabel('Cut Quality')
axes[0].set_ylabel('Count')
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: Average price by cut
sns.barplot(data=analysis_reset, x='cut', y='avg_price', 
            color='coral', alpha=0.7, ax=axes[1])
axes[1].set_title('Average Price by Cut')
axes[1].set_xlabel('Cut Quality')
axes[1].set_ylabel('Average Price ($)')
axes[1].grid(axis='y', alpha=0.3)

# Plot 3: Average carat by cut
sns.barplot(data=analysis_reset, x='cut', y='avg_carat', 
            color='green', alpha=0.7, ax=axes[2])
axes[2].set_title('Average Carat by Cut')
axes[2].set_xlabel('Cut Quality')
axes[2].set_ylabel('Average Carat')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```


```python

```
