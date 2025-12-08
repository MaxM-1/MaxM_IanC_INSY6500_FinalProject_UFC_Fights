# Graphing, 1/3

## Anscombe's Quartet

### Load

Load Anscombe's dataset, available in Seaborn package.


```python
import pandas as pd

# dataset of interest in seaborn
import seaborn as sns
anscombe = sns.load_dataset('anscombe')
```

### Explore

Have a look at the data.


```python
anscombe
```

But what does this look like? Pandas makes it easy to do simple visualizations with the `plot` method.


```python
ds1 = anscombe[anscombe['dataset'] == 'I']
plot1 = ds1.plot.scatter(x='x', y='y')
```

To draw all four we can simply loop through each dataset and generate a plot.


```python
for dataset in ['I', 'II', 'III', 'IV']:
    data = anscombe[anscombe['dataset'] == dataset]
    data.plot.scatter(x='x', y='y', title=f'Dataset {dataset}', figsize=(4,3))
```

Again we see:

- DS1 appears linear
- DS2 appears curvilinear
- DS3 appears linear with an outlier
- DS4 appears to have no relationship at all, except for one outlier

Here I've used the optional arguments `title` and `figsize` to customize the output a bit. There are **many** other options...


```python
help(anscombe.plot)
```

Four datasets. Overall stats?


```python
anscombe.describe()
```

How helpful is this? What would be better?

#### Start with Questions!

Unguided EDA is of limited utility [1]. Best to start with questions related to the data and/or the phenomena it measures / represents.

[1]: But sometimes just getting a sense of the data is the point → "unsupervised learning" (INSY 7130)

Can get we get those stats _grouped by_ dataset?


```python
anscombe.groupby('dataset').describe()
```

What do you see?

To get right to the heart of it, we can use the `agg` method to generate our own set of summary statistics for each column...


```python
anscombe.groupby('dataset').agg(['mean', 'std', 'var'])
```

What about x-y correlation by group?


```python
anscombe.groupby('dataset').apply(lambda df: df['x'].corr(df['y']), include_groups=False)
```

All correlations are ~0.816. This means if you fit a regression line, you'd get nearly identical equations for all four datasets (as shown in the slides).

Does this match your intuition about the data from the plots?

#### Spoiler Alert!

All four datasets have the same statistical properties despite having fundamentally different distributions!

**This is why we _must_ visualize.** Visualization is not an optional exercise to impress your reader. It is an essential step in the exploration process!

**The key insight:** Even sophisticated measures like correlation (r ≈ 0.816 for all four!) can completely mislead you without visualization. Dataset II needs a polynomial model, Dataset III needs outlier treatment, and Dataset IV... barely has any data relationship at all.

#### Sidebar - Correlation Cell

The previous code:

```python
anscombe.groupby('dataset').apply(lambda df: df['x'].corr(df['y']), include_groups=False)
```

...deserves a moment to unpack:

- `anscombe.groupby('dataset')` → perform aggregations that follow on each resulting group
- `apply(lambda df: df['x'].corr(df['y'])` → define a temporary function (lambda) that:
  - takes a dataframe (`df`)
  - returns the correlation between `x` and `y` columns
- `include_groups=False` → exclude the grouping column ('dataset') from `df`, passed to the lambda function

Instead of using `include_groups` we could select only the `x` and `y` columns:


```python
anscombe.groupby('dataset')[['x', 'y']].apply(lambda df: df['x'].corr(df['y']))
```

Either are equivalent to manually building a list of correlations and converting it to a dataframe with the dataset as the index:


```python
correlations = []
for dataset in ['I', 'II', 'III', 'IV']:
    data = anscombe[anscombe['dataset'] == dataset]
    corr = data['x'].corr(data['y'])
    correlations.append((dataset, corr))
    
pd.DataFrame(correlations, columns=['dataset', 'correlation']).set_index('dataset')
```

Or building a dictionary of `dataset:corr` k-v pairs and converting it to a series...


```python
correlations = {}
for dataset in ['I', 'II', 'III', 'IV']:
    data = anscombe[anscombe['dataset'] == dataset]
    correlations[dataset] = data['x'].corr(data['y'])
    
pd.Series(correlations, name='correlation')
```

The one-liner is more _idomatic_ to Pandas - more like the kind of Pandas you will see in the wild (pun intended).

### Matplotlib and Seaborn

####  Pandas is a matplotlib _Wrapper_

Pandas incorporates matplotlib and gives you a "friendly", higher-level _interface_ (aka _wrapper_) to its functions. This interface is called an API (application programming interface), which is a specification that describes the way two pieces of software "talk" to one another - the interface between them.


```python
type(plot1)
```

As you can see, the plot we originally created with Pandas is, in fact, a matplotlib object!

Pandas' `plot` method supports several important chart types, including:

- `scatter` for relationships
- `line` for trends
- `histogram` for univariate distributions
- `box` for distribution summary and outliers
- `bar` for categorical comparisons

You can do a lot with Pandas' plot functionality alone, but for complete control, you need matplotlib. For something in-between Pandas and matplotlib (plus some statistical goodies), there is Seaborn.

#### Matplotlib

For now, let's look at what matplotlib can do...


```python
import matplotlib.pyplot as plt
import numpy as np

# Create a 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Anscombe's Quartet: Same Stats, Different Data", 
             fontsize=16, fontweight='bold')

# Flatten axes for easier iteration
axes = axes.flatten()

# Plot each dataset
for i, dataset in enumerate(['I', 'II', 'III', 'IV']):
    ax = axes[i]
    data = anscombe[anscombe['dataset'] == dataset]
    
    # Scatter plot
    ax.scatter(data['x'], data['y'], s=30, alpha=0.6, color='steelblue')
    
    # Add regression line
    x = data['x']
    y = data['y']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), 'r--', linewidth=2, alpha=0.8)
    
    # Formatting
    ax.set_title(f'Dataset {dataset}', fontsize=12, pad=10)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2, 20)
    ax.set_ylim(2, 14)

plt.tight_layout()
plt.show()
```

#### Seaborn

Or with seaborn...


```python
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load the data
anscombe = sns.load_dataset('anscombe')

# Set style
sns.set_theme(style="darkgrid", palette="muted")

# Create FacetGrid
g = sns.FacetGrid(anscombe, col='dataset', col_wrap=2, 
                  height=4.5, aspect=1.2)

# Plot with regression
g.map_dataframe(sns.regplot, x='x', y='y',
                scatter_kws={'s': 20, 'alpha': 0.6, 'color': 'steelblue'},
                line_kws={'color': 'crimson', 'linewidth': 2.5, 'linestyle': '--'},
                ci=95)

# Add statistics to each plot
def add_stats(data, **kwargs):
    ax = plt.gca()
    x, y = data['x'], data['y']
    
    # Calculate statistics
    r, p = stats.pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    
    # Create text box
    stats_text = f'r = {r:.3f}\ny = {slope:.2f}x + {intercept:.2f}'
    ax.text(0.05, 0.95, stats_text, 
            transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5))

g.map_dataframe(add_stats)

# Titles
g.set_titles("Dataset {col_name}", size=15, weight='bold')
g.set_axis_labels("x", "y", size=13)
g.fig.suptitle("Anscombe's Quartet: Regression Analysis", 
               fontsize=17, fontweight='bold', y=1.01)

plt.tight_layout()
plt.show()
```

Seaborn is built on top of matplotlib but specializes in statistical graphics.

Notice how much less code we need for:

- Automatic faceting by categorical variable (`col='dataset'`)
- Built-in regression lines with confidence intervals
- Beautiful default styling
- Consistent formatting across subplots

Seaborn sits between Pandas plotting (fastest) and matplotlib (most control). It's perfect for EDA when you want both statistics and aesthetics.
