# Graphing, 3/3

Final(ish) part in our exploration of key graphing libraries.

- Use Seaborn for statistical visualization (2/2)
- Splashes of the Matplotlib API



```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

diamonds = sns.load_dataset('diamonds')
df = diamonds.sample(n=5000, random_state=42)
df.head()
```

## Seaborn: Statistical Graphics

Seaborn builds on matplotlib to provide:

- Statistical visualizations (regression, distributions, confidence intervals)
- Beautiful defaults with minimal code
- Easy aesthetic mappings (`hue`, `size`, `style`)
- Grammar of Graphics concepts in practice

It returns matplotlib objects, so you can customize further if needed. Let's see it in action.

### Seaborn's API Pattern

Consistent API across all plot types:

```python
sns.function_name(
    data=df,           # 1. Your DataFrame
    x='column1',       # 2. What goes where
    y='column2',
    hue='category',    # 3. Aesthetic mappings (optional)
    size='continuous',
    style='category2'
)
```

The key difference from Pandas:

- Pandas: `df.plot.scatter(x='col1', y='col2')`
- Seaborn: `sns.scatterplot(data=df, x='col1', y='col2')`

Separating the data from the aesthetics enables more powerful mappings.

#### Function families

- Relational: `scatterplot`, `lineplot`, `regplot`
- Distributional: `histplot`, `kdeplot`, `boxplot`, `violinplot`
- Categorical: `barplot`, `countplot`, `stripplot`, `swarmplot`

Choose based on your variable types and question.

Notice how the parameters map to Grammar of Graphics components:

- `data` = your dataset
- `x`, `y` = aesthetic mappings (position)
- `hue`, `size`, `style` = aesthetic mappings (color, size, shape)
- Function name (e.g., `scatterplot`) = geometry
- Built-in statistics (e.g., regression, aggregation) = statistical transformation

We'll cover these concepts in action shortly, but first we need to discuss how categorical variables are ordered.

### Categorical Variables

The diamonds dataset has data that is categorical and ordered in nature:

- `cut`: Fair < Good < Very Good < Premium < Ideal
- `color`: D > E > F > G > H > I > J (D=best/colorless to J=worst)
- `clarity`: I1 < SI2 < SI1 < VS2 < VS1 < VVS2 < VVS1 < IF (worst to best)

But it is important to ensure this is correctly set. To check the data type, categories, and if it is ordered or not:


```python
print(df['cut'].dtype)
print(df['cut'].cat.categories)
print(df['cut'].dtype.ordered)
```

`cut` is a categorical variable with the allowable values set by its `Index`, which, in this case is *unordered*. Let's check the others:


```python
print(df['color'].dtype.ordered)
print(df['clarity'].dtype.ordered)
```

None are ordered by default. We will correct that shortly!

#### Ordered vs Unordered

The following cell illustrates the difference between ordered and unordered categories.


```python
import pandas as pd

# Create unordered categorical
unordered = pd.Categorical(
    ['Medium', 'High', 'Low', 'Medium', 'High'],
    categories=['Low', 'Medium', 'High'],
    ordered=False
)

# Create ordered categorical  
ordered = pd.Categorical(
    ['Medium', 'High', 'Low', 'Medium', 'High'],
    categories=['Low', 'Medium', 'High'],
    ordered=True
)

print("Unordered:")
print(f"  ordered attribute: {unordered.ordered}")
print(f"  categories: {unordered.categories}")

print("\nOrdered:")
print(f"  ordered attribute: {ordered.ordered}")
print(f"  categories: {ordered.categories}")  
```

#### Internal Representation

Under the hood, all categoricals are represented by integer codes:


```python
print("--- Internal Representation ---")
print(f"Unordered codes: {unordered.codes}")
print(f"Ordered codes:   {ordered.codes}")
print("\nBoth use the SAME integer encoding!")
print(f"  'Low' -> 0, 'Medium' -> 1, 'High' -> 2")

print("\n--- The ONLY difference is the ordered flag ---")
print(f"Unordered.ordered: {unordered.ordered}")
print(f"Ordered.ordered:   {ordered.ordered}")
```

#### Categorical Advantages

Advantages of categorical variables over strings:

- fixed set of acceptable values
- memory savings
- many techniques require categorical representation (regress on integers, not category names)


```python
print("\n--- Memory efficiency ---")
import sys
strings = ['Medium', 'High', 'Low'] * 1000
categorical = pd.Categorical(strings * 1000)
print(f"3000 strings as list: ~{sys.getsizeof(strings * 1000):,} bytes")
print(f"3000 strings as categorical: ~{sys.getsizeof(categorical):,} bytes")
print("(Categorical stores: int codes + category list)")
```

#### Comparing Categorical Variables

**Important note**:

Scalar indexing of a categorical returns a **string**, so comparisons / ordering is lost:


```python
print(f"ordered[0]:\n{ordered[0]}")
print(f"\ntype: {type(ordered[0])}")
print(f"String comparison, 'Low' < 'Medium': {ordered[0] < ordered[1]}")
```

The comparison is string based, not category based.

Array slicing preserves the categorical, so ordering is preserved:


```python
print(f"ordered[[0]]:\n{ordered[[0]]}")
print(f"\ntype: {type(ordered[[0]])}")
print(f"Categorical comparison, 'Low' < 'Medium': {(ordered[[0]] < ordered[[1]])[0]}")
```

Array comparisons work correctly, e.g. for Boolean masks:


```python
print(f"ordered:\n{ordered}")
print(f"\nordered < 'High':")
print(ordered < 'High')  # Returns boolean array
print(f"Count less than High: {(ordered < 'High').sum()}")
```

Differences summarized:

| Feature | Unordered | Ordered |
|---------|-----------|---------|
| `.cat.ordered` | `False` | `True` |
| Comparisons (`<`, `>`) | ❌ Error | ✅ Works |
| Sorting | Arbitrary | By defined order |
| Plotting default | Alphabetical or appearance | Categorical order |

#### Creating / Converting Categorical Variables

To convert string / object data into categories, where order doesn't matter, you can simply use `astype`:

```python
df['city'] = df['city'].astype('category')
```

Pandas will infer the categories, and the Index will be alphabetical but the type is unordered.

To convert a string / object data into ordered categories, you must explicitly define the order:


```python
df['cut'] = pd.Categorical(
    df['cut'],
    categories=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
    ordered=True
)

print(df['cut'].dtype)
print(df['cut'].cat.categories)
print(df['cut'].dtype.ordered)
```

This approach is best even if unordered categories exist, to explicitly control the order.


```python
df['color'] = pd.Categorical(
    df['color'],
    categories=list('JIHGFED'),  # creates list of string elements in increasing color quality ['J', 'I', ...]
    ordered=True
)

print(df['color'].dtype)
print(df['color'].cat.categories)
print(df['color'].dtype.ordered)

df['clarity'] = pd.Categorical(
    df['clarity'],
    categories=['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'],
    ordered=True
)

print(df['clarity'].dtype)
print(df['clarity'].cat.categories)
print(df['clarity'].dtype.ordered)
```

### Quick Tour of Seaborn Plot Types

#### For Distributions

##### Histograms with KDE

Kernel Density Estimation (KDE) gives a smoothed estimate of the underlying distribution. Useful for seeing the shape beyond the binning artifacts of histograms.

Seaborn uses the **Freedman-Diaconis rule** for automatic binning, which is more robust to outliers and skewed data than Pandas' method. It calculates optimal bin width based on:

$$\text{bin width} = 2 \times \text{IQR} \times n^{-1/3}$$

This adapts to your data's spread and sample size.


```python
sns.histplot(data=df, x='carat', kde=True)
```

Notice that, by default, Seaborn does not show the grid. It is considered best practice to reduce the clutter in your charts, and the grid isn't normally necessary.

##### Better Box Plots

Automatic grouping and statistical summaries (quartiles, median, outliers). Add color to show the ordering of cut quality.


```python
sns.boxplot(data=df, x='cut', y='price', 
            hue='cut',
            palette='viridis',
            legend=False)
```

The viridis palette is designed to be friendly to those with color-blindness. Seaborn / matplotlib offer several others.

By default, Seaborn handles ordering of categorical variables as follows:

1. If pandas column is **categorical with order** → uses that order
2. If just **string/object type** → uses order of appearance or alphabetical
3. Can override with `order` parameter


```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Default: uses pandas categorical order
sns.countplot(data=df, x='cut', ax=axes[0], legend=False)
axes[0].set_title('Default: Pandas Order')

# Override with seaborn order parameter (reversed)
sns.countplot(data=df, x='cut', 
              order=['Ideal', 'Premium', 'Very Good', 'Good', 'Fair'],
              ax=axes[1], legend=False)
axes[1].set_title('Reversed with order=')

# Sort by frequency
sns.countplot(data=df, x='cut',
              order=df['cut'].value_counts().index,
              ax=axes[2], legend=False)
axes[2].set_title('Sorted by Frequency')

plt.tight_layout()
```

It is considered best practice to set categorical order in pandas during data cleaning, then all downstream plots use it automatically. Use Seaborn's `order` parameter only for one-off visualizations.

##### Violin Plots

Combines box plot statistics with full distribution shape. Shows density at different values - wider sections indicate more observations.


```python
sns.violinplot(data=df, x='cut', y='price')
```

Notice how Premium and Ideal cuts show slight bimodal distributions (two peaks), suggesting distinct price clusters within those grades.

#### Relationship Plots

##### Scatter with Regression

Automatically fits a linear regression and shows 95% confidence interval (shaded region).

This combines two "geoms" (h/t gg) in one plot - point and line. To set the properties of each we use the `scatter_kws` and `line_kws` (keywords) parameters to provide separate dictionaries of settings.


```python
sns.regplot(data=df, x='carat', y='price',
            scatter_kws={'alpha': 0.3},
            line_kws={'color': 'red'})
```

Notice how the confidence interval widens at the extremes - we have less data there, so more uncertainty.

##### Line Plots with Uncertainty

Line plots are best suited for ordered data. Let's create price bins and see how average carat varies across them - demonstrating the Filter → Aggregate → Plot workflow.


```python
# Pandas workflow: bin prices, then aggregate by cut
df['price_bin'] = pd.cut(df['price'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

sns.lineplot(data=df, x='price_bin', y='carat', 
             hue='cut', marker='o', err_style='bars')
```

#### Categorical Plots

##### Bar Plots with Error Bars

Automatic aggregation (mean by default) with error bars showing variability.

Here we compute mean price by cut, with standard deviation bars.


```python
sns.barplot(data=df, x='cut', y='price', 
            estimator='mean', errorbar='sd',
            hue='cut', palette='coolwarm', legend=False)
```

 Notice Premium has highest average - because it includes more large diamonds, not better quality per se.

##### Count Plots

Quick frequency visualization - no aggregation needed, just counts observations.


```python
sns.countplot(data=df, x='color', 
              hue='color', palette='Spectral', legend=False)
```

Color grades show normal distribution around G-H (middle quality).

##### Strip / Swarm Plots

Bar plot style, but shows individual data points, not just aggregates. Strip adds random horizontal "jitter" to reduce overplotting.

Let's filter to smaller diamonds for clarity.


```python
# Filter → Plot workflow
small_diamonds = df[df['carat'] < 1.0]

sns.stripplot(data=small_diamonds, x='clarity', y='carat', 
              hue='clarity', palette='Set1', alpha=0.5, legend=False)
```

Ack! Choose your color palettes wisely!

Swarm plot is the same but dots don't overlap. This limits the number of points you can plot without warnings.


```python
sns.swarmplot(data=df.sample(100), x='clarity', y='carat')
```

### Multiple Aesthetic Mappings

Seaborn makes it easy to apply concepts from the Grammar of Graphics, mapping multiple variables to different visual properties.

The following plot explores the relationship between carat and price while encoding additional information through color, size, and shape:


```python
sns.scatterplot(data=df.sample(1000), 
                x='carat', y='price',       # Position aesthetics
                hue='cut',                  # Color aesthetic (categorical)
                size='depth',               # Size aesthetic (continuous)
                style='color',              # Shape aesthetic (categorical)
                alpha=0.6,
                sizes=(20, 200))
```

### Faceting

Per the component of the same name in the grammar of graphics. Called subplots in matplotlib.

Allows us to see how a relationship between two attributes changes based on a third.


```python
# define the grid first
g = sns.FacetGrid(data=df, col='clarity', col_wrap=4, height=2)

# then add plots, specify attributes
g.map_dataframe(sns.scatterplot, x='carat', y='price')
g.set_titles("Category: {col_name}")
```

Use faceting when:

- there are too many categories for other aesthetics, e.g. color
- you want / need to see within-group patterns
- relationships are multi-dimensional

#### Other Ways to Combine Plots

In addition to sns' faceting, you can utilize matplotlib subplots or automatically overlay plots.

When to use each:

- `FacetGrid`: Systematic small multiples across categorical variable
- matplotlib `subplots`: Custom layouts, different plot types side-by-side
- Overlay: Multiple layers on same axes (like histogram + rug plot)
- Two axes: similar to overlay but using the left and right axes to represent different values

Let's revisit the histogram example and explore auto vs manual binning and show multiple distributions.

- Auto-binning is usually good, but you might want:
  - More detail (more bins) to see fine structure
  - Less noise (fewer bins) for clearer patterns
  - Consistent bins across multiple plots for comparison
  - Round numbers for interpretability



```python
# create matplotlib plot object with subplots, get figure and axes for access
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Auto-binning (Seaborn's smart default)
sns.histplot(data=df, x='carat', kde=True, ax=axes[0])
axes[0].set_title('Auto-binning (Freedman-Diaconis)')

# Too few bins - loses detail
sns.histplot(data=df, x='carat', bins=10, kde=True, ax=axes[1])
axes[1].set_title('Too Few Bins (10)')

# Multiple distributions with hue
sns.histplot(data=df, x='carat', hue='cut', kde=True, 
             bins=30, ax=axes[2], alpha=0.5)
axes[2].set_title('Multiple Distributions by Cut')

plt.tight_layout()
```

Alternatively, sequential sns plots are overlayed. The following code adds a "rug" plot at the base of the histogram. This gives you another way to visualize the distribution.


```python
sns.histplot(data=df, x='carat', kde=True, bins=40)
sns.rugplot(data=df, x='carat', height=0.05, alpha=0.3)
plt.title('Histogram + Rug Plot (individual points at bottom)')
```

The twin axes approach also relies on matplotlib subplots.


```python
fig, ax1 = plt.subplots(figsize=(6, 4))

# Histogram on left y-axis
sns.histplot(data=df, x='carat', bins=40, stat='density',
             kde=False, ax=ax1, color='steelblue', alpha=0.6)
ax1.set_ylabel('Density', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')

# Cumulative distribution on right y-axis
ax2 = ax1.twinx()
sns.histplot(data=df, x='carat', bins=40, stat='density',
             cumulative=True, element='step', fill=False,
             color='red', linewidth=2, ax=ax2)
ax2.set_ylabel('Cumulative Probability', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(0, 1)

plt.title('Histogram with Cumulative Distribution')
plt.tight_layout()
```

### When to Use Seaborn Plotting

Use Seaborn When:

- ✓ Exploring relationships between variables
- ✓ Want statistical transformations (regression, CI)
- ✓ Need beautiful defaults quickly
- ✓ Using categorical variables extensively
- ✓ Want faceting
- ✗ Need very custom layouts
- ✗ Need fine-grained control


```python

```
