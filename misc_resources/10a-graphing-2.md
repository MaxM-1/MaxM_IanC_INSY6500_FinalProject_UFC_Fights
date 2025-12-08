# Graphing, 2/3

Build a systematic approach to visualization.

- Master Pandas' plotting capabilities for quick EDA
- Understand the Grammar of Graphics as a thinking framework
- Use Seaborn for statistical visualization (1/2)

Seaborn comes with a number of datasets:


```python
import pandas as pd
import seaborn as sns

sns.get_dataset_names()
```

Each dataset is described here: [Seaborn Data repo](https://github.com/mwaskom/seaborn-data)

For this exercise, we'll use [diamonds](https://ggplot2.tidyverse.org/reference/diamonds.html) - 54,000 observations of diamond characteristics and prices. This is real retail data, so it has all the features you'll see in industry: skewed distributions, ordinal quality grades, multicollinearity between size metrics, and non-linear price relationships. Perfect for learning EDA because it forces us to think critically about what we're visualizing.


```python
diamonds = sns.load_dataset('diamonds')
```


```python
diamonds.head()
```

This data is ***tidy***: each row is an observation, columns are variables, and the table is about one "observational unit" (thing being measured). More about this important point later.

The columns are:

- price: price in US dollars ($326–$18,823)
- carat: weight of the diamond (0.2–5.01)
- cut: quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- color: diamond colour, from D (best) to J (worst)
- clarity: a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
- x: length in mm (0–10.74)
- y: width in mm (0–58.9)
- z: depth in mm (0–31.8)
- depth: total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43–79)
- table: width of top of diamond relative to widest point (43–95)



```python
diamonds.info()
```


```python
diamonds.describe()
```

You can do more initial numerical exploration, but, as we saw last time, visualization is essential.

Pandas is perfect for quick early visualization.

## Plotting in Pandas

### The `.plot` API

API = Application Programming Interface, the language that you use to "talk" to software.

When using libraries, it refers to the function "signatures" - the names, parameters, return values, and overall structure of how you interact with the code.

For plotting in Pandas, it takes the form of the `plot` method:

```python
df.plot(kind, x, y, ...)
```

Here, `df` is the dataframe (or series) of interest, `kind` specifies the plot type, and `x` and `y` specify the columns to use for each axis. For more detail, you can refer to the [Pandas user guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html) or [reference](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html). The latter is also available via help:

```text
>>> help(df.plot)

...

 |  Parameters
 |  ----------
 |  data : Series or DataFrame
 |      The object for which the method is called.
 |  x : label or position, default None
 |      Only used if data is a DataFrame.
 |  y : label, position or list of label, positions, default None
 |      Allows plotting of one column versus another. Only used if data is a
 |      DataFrame.
 |  kind : str
 |      The kind of plot to produce:
 |
 |      - 'line' : line plot (default)
 |      - 'bar' : vertical bar plot
 |      - 'barh' : horizontal bar plot
 |      - 'hist' : histogram
 |      - 'box' : boxplot
 |      - 'kde' : Kernel Density Estimation plot
 |      - 'density' : same as 'kde'
 |      - 'area' : area plot
 |      - 'pie' : pie plot
 |      - 'scatter' : scatter plot (DataFrame only)
 |      - 'hexbin' : hexbin plot (DataFrame only)

...
```

Beyond those parameters, we will find the following most useful:

- `figsize=(width, height)` - plot size
- `title='My Title'` - adding titles
- `xlabel`, `ylabel` - axis labels
- `alpha=0.5` - transparency
- `color='steelblue'` - color control
- `grid=True` - gridlines

Simple example:


```python
diamonds['price'].plot.hist(bins=50, title='Price Distribution')
```

For this lecture we will work with a random sample of the dataset.

In industry, you'll often work with millions of rows. Sampling for visualization is standard practice - you explore with a sample, then validate with the full dataset.


```python
df = diamonds.sample(n=5000, random_state=42)

print(f"Full dataset: {len(diamonds):,} rows")
print(f"Sample for visualization: {len(df):,} rows")
```

Note that it is common to name your working dataset `df`, also to preserve the original data.

### Pandas Plotting Workflow

Use Pandas when you just need to see something quickly.

Filter → Group → Aggregate → Plot

This workflow is good enough for the majority of EDA.

#### Example 1: Distribution of a Continuous Variable

Question: What is the distribution of diamond prices?

Variable type: `price` is continuous

Appropriate plot: Histogram (shows frequency distribution)


```python
# Simple histogram
df['price'].plot.hist(bins=50, 
                      figsize=(8, 4.5),
                      title='Distribution of Diamond Prices',
                      xlabel='Price ($)',
                      ylabel='Frequency',
                      color='steelblue',
                      alpha=0.7,
                      grid=True)
```

Observation: Heavily right-skewed - most diamonds are relatively inexpensive, with a long tail of high-priced diamonds.

#### Example 2: Relationship Between Two Continuous Variables

Question: How does carat weight relate to price?

Variable types: `carat` and `price` are both continuous

Appropriate plot: Scatter plot (shows relationships)


```python
# Scatter plot
diamonds.plot.scatter(x='carat', y='price',
                figsize=(8, 5),
                title='Price vs Carat Weight',
                xlabel='Carat',
                ylabel='Price ($)',
                alpha=0.1,
                color='darkblue',
                grid=True)

```

Observation: Clear positive relationship, but non-linear - price increases exponentially with carat.

*Why alpha? What happens with the full dataset?*

#### Example 3: Comparing Continuous Variable Across Ordinal Categories

Question: How does price vary by cut quality?

Variable types: `cut` is ordinal (Fair < Good < Very Good < Premium < Ideal), `price` is continuous

Appropriate plot: Box plot (shows distribution by group)

Workflow: Filter (if needed) → Group by `cut` → Plot distribution

> Note: boxplots are a bit clunky in Pandas. We'll see a cleaner approach with Seaborn.


```python
df.plot.box(column='price', by='cut',
            figsize=(10, 6),
            title='Price Distribution by Cut Quality',
            ylabel='Price ($)',
            xlabel='Cut Quality (random order?)',
            grid=True)
```

Observation: Surprisingly, "Ideal" cut diamonds don't have the highest median price. This is likely because cut quality is independent of size (carat), and size is a stronger price driver.

#### Example 4: Aggregation and Categorical Comparison

Question: What is the average price for each cut quality?

Variable types: `cut` is ordinal, `price` is continuous

Appropriate plot: Bar plot (shows comparisons of aggregated values)

Workflow: Group → Aggregate (mean) → Plot

> Note: `observed='false'` tells `groupby` to use all defined categories, even if some are empty. This was the default behavior but is being deprecated, so I've set it explicitly here (and elsewhere in the notebook).


```python
df.groupby('cut', observed='false')['price'].mean().plot.bar(
    figsize=(8, 5),
    title='Average Price by Cut Quality',
    xlabel='Cut Quality (decreasing)',
    ylabel='Average Price ($)',
    color='steelblue',
    alpha=0.5,
    grid=True)
```

Observation: Average prices are relatively similar across cuts, with Premium and Ideal slightly higher. The ordering isn't as dramatic as you might expect.

*How is the ordering of the groups done? How does it influence your interpretation of the chart?*

#### Example 5: Filtering and Comparing

Question: For diamonds over 2 carats, how do prices compare across cut qualities?

Variable types: `cut` is ordinal, `price` is continuous

Workflow: Filter → Group → Aggregate → Plot


```python
# Filter → Group → Aggregate → Plot
large_diamonds = df[df['carat'] > 2]

large_diamonds.groupby('cut', observed='false')['price'].mean().plot.bar(
    figsize=(8, 5),
    title='Average Price by Cut (Diamonds > 2 Carats)',
    xlabel='Cut Quality',
    ylabel='Average Price ($)',
    color='darkgreen',
    alpha=0.8,
    grid=True)

# Show sample sizes
print("\nSample sizes:")
print(large_diamonds['cut'].value_counts().sort_index())

```

Observation: For large diamonds, the pattern changes - Premium and Ideal cuts command higher average prices. Sample sizes vary significantly by cut.

#### Example 6: Discrete Variable - Frequency Count

Question: How many diamonds are there of each color grade?

Variable type: `color` is ordinal/categorical (D = best, J = worst)

Appropriate plot: Bar plot (shows counts)

Workflow: Group → Count → Plot


```python
df['color'].value_counts().sort_index()
```


```python
# Value counts automatically aggregates
df['color'].value_counts().sort_index().plot.bar(
    figsize=(8, 5),
    title='Number of Diamonds by Color Grade',
    xlabel='Color Grade (D=Best, J=Worst)',
    ylabel='Count',
    color='coral',
    alpha=0.8,
    grid=True)
```

Observation: Most diamonds in the dataset are in the middle color grades (G, H, I). Best (D-F) and worst (J) grades are less common.

The workflow is consistent:

1. Start with a question
2. Identify variable types
3. Choose appropriate plot
4. Filter/group/aggregate as needed
5. `.plot`

### When to Use Pandas Plotting

**Use Pandas when:**

- ✓ Quick exploratory look
- ✓ Simple single or two-variable plots
- ✓ You just need to see something fast
- ✗ Need statistical layers (regression, confidence intervals)
- ✗ Need complex multi-plot layouts
- ✗ Creating publication-quality figures

Pandas is perfect for initial EDA. After that, you'll need more powerful tools - which we'll explore next with the Grammar of Graphics and Seaborn.

## The Grammar of Graphics (Wickham, 2010 and 2014)

### Tidy Data

As we mentioned before, and alluded to repeatedly in this class, data should be *tidy*:

- every table is one "observational unit" (thing)
- every row is an observation
- every column is a variable

This enables the column-based analysis tools that we rely on. This idea was formalized by Hadley Wickham, the creator of `ggplot` and much of the `tidyverse` ecosystem that has been instrumental in making R a popular language for data analysis and visualization.

The idea of tidy data is sensible but most data is not tidy. Common examples of non-tidy data include:

1. Column headers are values, not variable names

Non-tidy:

```text
| Country | 2020 | 2021 | 2022 |
|---------|------|------|------|
| USA     | 100  | 110  | 120  |
| Canada  | 80   | 85   | 90   |
...
```

Tidy:

```text
| Country | Year | Value |
|---------|------|-------|
| USA     | 2020 | 100   |
| USA     | 2021 | 110   |
| USA     | 2022 | 120   |
| Canada  | 2020 | 85    |
...
```

Notice how tidy data tends to be *long*, not *wide*.

2. Multiple variables in one column

Non-tidy:

```text
| ID | Gender_Age |
|----|------------|
| 1  | M_25       |
| 2  | F_30       |
| 3  | M_42       |
...
```

Tidy:

```text
| ID | Gender | Age |
|----|--------|-----|
| 1  | M      | 25  |
| 2  | F      | 30  |
| 3  | M      | 42  |
...
```

Each variable should be its own column. In this case the tidy data is wider.

3. Variables in both rows and columns (pivot table format)

Non-tidy:

```text
| Date       | Temp_Morning | Temp_Evening | Humidity_Morning | Humidity_Evening |
|------------|--------------|--------------|------------------|------------------|
| 2024-01-01 | 45           | 52           | 65               | 70               |
| 2024-01-02 | 43           | 50           | 68               | 72               |
...
```

Tidy:

```text
| Date       | Time    | Temp | Humidity |
|------------|---------|------|----------|
| 2024-01-01 | Morning | 45   | 65       |
| 2024-01-01 | Evening | 52   | 70       |
| 2024-01-02 | Morning | 43   | 68       |
| 2024-01-02 | Evening | 50   | 72       |
...
```

Variable names should not encode values.

4. Multiple observation types in one column

Non-tidy:

```text
| ID | Measurement  |
|----|--------------|
| 1  | Height: 5.8  |
| 1  | Weight: 180  |
| 1  | Age: 30      |
| 2  | Height: 6.1  |
...
```

Tidy:

```text
| ID | Height | Weight | Age |
|----|--------|--------|-----|
| 1  | 5.8    | 180    | 30  |
| 2  | 6.1    | 195    | 28  |
...
```

Each measurement type should be its own variable.

5. Repeated measures as columns (wide format)

Non-tidy:

```text
| Subject | Trial1 | Trial2 | Trial3 |
|---------|--------|--------|--------|
| A       | 12.3   | 11.8   | 12.1   |
| B       | 15.2   | 14.9   | 15.0   |
...
```

Tidy:

```text
| Subject | Trial | Value |
|---------|-------|-------|
| A       | 1     | 12.3  |
| A       | 2     | 11.8  |
| A       | 3     | 12.1  |
| B       | 1     | 15.2  |
| B       | 2     | 14.9  |
| B       | 3     | 15.0  |
...
```

Again, tidy data tends to be *long* rather than *wide*.
```

Notice how all our Pandas plotting examples worked smoothly because the diamonds dataset is tidy:

- `df['price'].plot.hist()` - works because price is one column
- `df.plot.scatter(x='carat', y='price')` - works because x and y are separate columns
- `df.groupby('cut')['price'].mean()` - works because cut is one column, price is another

If the data weren't tidy, we'd have to reshape it first. Tidy data makes visualization (and analysis) straightforward.

### The Core Philosophy

A visualization is not a "chart type" - it's a mapping from data to visual properties, composed of layers.

Instead of asking *Should I make a scatter plot or bar chart?*, ask:

1. What *variables* am I exploring?
2. How should they map to *visual channels*?
3. What *geometric representation* clarifies the pattern?

### The Five Key Components

#### 1. Data

The tidy dataset you're working with

- Each row = observation
- Each column = variable

#### 2. Aesthetic Mappings

How data variables map to visual properties:

|Visual 'Dimension'|Example Mapping|
|---|---|
|Position (x, y)|time → x, sales → y|
|Color|category → hue, value → saturation|
|Size|population → point size|
|Shape|species → marker shape|
|Transparency|confidence → alpha|

Key insight: The same data can be mapped different ways to reveal different patterns. The most important concept?

#### 3. Geometric Objects

The visual marks representing data:

- Points (scatter)
- Lines (trends)
- Bars (comparisons)
- Areas (distributions)

You can layer multiple geometries on the same aesthetic mapping (e.g., points + smoothed line).

#### 4. Statistical Transformations

Data transformations before plotting:

- Raw values (identity)
- Counts/binning (histograms)
- Aggregation (means, medians)
- Smoothing (regression lines, loess)

#### 5. Facets

Breaking into small multiples by categorical variables

- "Show me this pattern separately for each category"
- Anscombe's Quartet is faceted by `dataset`

### Additional Concepts

Scales: Control how data maps to aesthetics

- Linear vs log scales
- Color palettes
- Axis limits and breaks

Coordinate Systems:

- Cartesian (standard)
- Polar (circular)
- Map projections

### Systematic Application

Traditional Approach:

"I need to show sales by region over time... maybe a grouped bar chart? Or multiple line plots? Let me Google 'matplotlib grouped bar chart'..."

Grammar Approach:

1. Variables: time (continuous), sales (continuous), region (categorical)
2. Mappings: time → x, sales → y, region → color
3. Geometry: lines (shows trends better than bars for time series)
4. Result: Line plot with one line per region

The tool choice(s) becomes (more) "obvious," especially when using Seaborn (or ggplot).

Seaborn (partial implementation):


```python
mpg = sns.load_dataset('mpg')

sns.scatterplot(
    data=mpg,          # 1. Data
    x='horsepower',    # 2. Aesthetic: position
    y='mpg',           # 2. Aesthetic: position  
    hue='cylinders',   # 2. Aesthetic: color
    size='weight',     # 2. Aesthetic: size
    style='origin',    # 2. Aesthetic: shape
    legend=True        # hide legend for now
)                      # 3. Geometry: points (scatterplot)
```

### The Mental Model

1. What questions am I asking?
   - Distribution? → one variable
   - Relationship? → two+ variables
   - Comparison? → groups
   - Change? → time
2. What mappings answer this?
   - Position (x, y) - most important, highest precision
   - Color - good for categories
   - Size - good for magnitude
   - Facets - great for categories when color isn't enough
3. What geometry reveals the pattern?
   - Points → relationships, individual values
   - Lines → trends, connections
   - Bars → comparisons, counts
   - Boxes/violins → distributions
4. What statistics help?
   - Raw data vs aggregated
   - Trends/smoothing
   - Confidence intervals

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


```python

```
