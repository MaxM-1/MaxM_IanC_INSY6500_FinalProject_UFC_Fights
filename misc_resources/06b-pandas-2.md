# More Pandas

Note: Parts of this notebook is adapted from chapter 5 of McKinney, which is available in HTML format here: [Pandas Basics](https://wesmckinney.com/book/pandas-basics).

Other portions draw from VanderPlas, chapters 13-16.


```python
import numpy as np
import pandas as pd

# sample data from previous lecture
res_1 = pd.Series([11.8, 30., 4.2, 3.4],
                 index=["Orlando", "Auburn", "Atlanta", "Birmingham"])

data = {"state": ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada"],
            "year": [2000, 2001, 2002, 2001, 2002, 2003],
            "pop": [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)
```

## Diagnostics

Series and DataFrame objects share a set of attributes / methods that are useful for getting to know your data.

### Structure

The `ndim` and `shape` attributes describe the structure of the data.


```python
# for a dataframe
print(frame)
print("\nndim:", frame.ndim)
print("shape:", frame.shape)
```


```python
# for a series
print(res_1)
print("\nndim:", res_1.ndim)
print("shape:", res_1.shape)
```

**Important Note:** a Series is not the same as a one-dimensional DataFrame. A Series is a one-dimensional labeled array. A DataFrame is **always** two-dimensional, even when it contains only one column or row.


```python
df = pd.DataFrame([1, 2, 3], index=['a', 'b', 'c'])
print(df)
print("\nndim:", df.ndim)
print("shape:", df.shape)
```

### Data Types

The `dtypes` method reports the types of data present in both Series and DataFrames.


```python
# dataframes have one type per column
print(frame.dtypes)
```


```python
# series is homogenous, all values must share the same type
print(res_1.dtypes)
```

### Head and Tail

The `head` and `tail` methods can be used to inspect the first / last 5 rows, respectively. Use the `n` parameter to set the number of rows.


```python
# check the first 5 rows of a DataFrame
print(frame.head())
```


```python
# check the last 3 rows of a Series
res_1.tail(n=3)
```

### Info

Get general information about the data structure, including object types and null values.


```python
frame.info()
```


```python
res_1.info()
```

## Indexing with `.loc[]` and `.iloc[]`

We've seen a few approaches to accessing the elements of a Series and DataFrame, along with some cautions / recommendations. This can be a sticky topic, so we offer the following recommendations:

- When accessing named **columns** of a DataFrame, use the `df['col_name']` approach described above.
- In **all other cases**, use `loc` or `iloc` as described below.

While `loc` and `iloc` don't offer the most concise notation, that disadvantage is more than offset by the consistency and explicit nature of this approach. It also aligns with best practices and the direction of Pandas development.

Note the use of square brackets to suggest the indexing / slicing syntax. `loc` and `iloc` are special *indexer attributes*, not functions, which are called with parentheses, not brackets.


```python
populations = {
    "Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6},
    "Nevada": {2001: 2.4, 2002: 2.9},
    "Texas": {2000: 8.4, 2001: 8.8}
}

frame3 = pd.DataFrame(populations)

print(frame3)
```

### `.loc[]` for Label-Based Access

The `loc` method is available for both Series and DataFrames. It provides a consistent way to access labeled rows and/or columns. To use it, you must specify an index or slice for each axis.


```python
# extract the first column by name as a series
# [all rows, column named "Ohio"]
frame3.loc[:, "Ohio"]
```

In this case, it is more concise and explicit to use the `df['col_name']` syntax as recommended above.


```python
frame3["Ohio"]
```

In **all other cases**, `loc` is recommended.


```python
# extract the first row by name as a series
# [row named 2000, all columns]
frame3.loc[2000, :]
```


```python
# extract a single element
# [row named 2000, column named "Texas"]
frame3.loc[2000, "Texas"]
```

When you select a single element using `.loc[row, col]`, pandas returns a NumPy scalar - in this case `np.float64(8.4)`. This is because pandas stores its numerical data using NumPy's data types under the hood.

While this might look different from a regular Python float like 8.4, you can use it the same way in calculations. As seen in the following example, it is equivalent:


```python
print(frame3.loc[2000, "Texas"] / 2)
```


```python
# extract a range by name
# row named 2001, cols "Ohio" thru "Nevada" - inclusive; closed interval, i.e., []
frame3.loc[2001, "Ohio":"Nevada"]
```

**Important Note:** unlike slices in base Python (or even when using `iloc` as we will see), slices in `loc` are **inclusive** of the end point. In the previous example the column data for "Nevada" was included in the output.

### `.iloc[]` for Integer Position-Based Access

The `iloc` method is also available for both Series and DataFrame objects. It provides a consistent way to access data by the numerical indicies. As with `loc`, you must specify an index or slice for each axis.

The following examples are analogous to the label-based ones above.


```python
# first column
# [all rows, column 0]
frame3.iloc[:, 0]
```


```python
# first row
# [row 0, all columns]
frame3.iloc[0, :]
```


```python
# single element
# [row 0, column 2]
frame3.iloc[0, 2]
```


```python
# range by position
# [row 1, columns 0 and 1] - exclusive of endpoint! half-open interval, i.e., [)
frame3.iloc[1, 0:2]
```

Note that `iloc` slices follow base Python, where the endpoint is **exclusive**. The slice above starts at column zero and goes up to, *but does not include*, the second column.

### One Last Exception

While *indexing* refers to columns, *slicing* refers to rows.


```python
area = pd.Series({'California': 423967, 'Texas': 695662,
                  'Florida': 170312, 'New York': 141297,
                  'Pennsylvania': 119280})

pop = pd.Series({'California': 39538223, 'Texas': 29145505,
                 'Florida': 21538187, 'New York': 20201249,
                 'Pennsylvania': 13002700})

data = pd.DataFrame({'area':area, 'pop':pop})
print(data)
```


```python
# slicing refers to rows; inclusive for labels
print(data['Texas':'New York'])
```


```python
# same when slicing by index, exclusive for index
print(data[1:3])
```

This is just something you have to memorize, I'm afraid.

### Indexed Assignment

Any of the indexing methods described above (and others available) can be used to modify the object.


```python
print(data)
```


```python
# add a column via index assignment
data["density"] = data["pop"] / data["area"]
print(data)
```


```python
data.loc["California", "density"] = 90
print(data)
```

## Missing Data

As we've seen, `NaN` (Not A Number) is the primary way that Pandas (and NumPy, from which it was inherited) represents missing values. You will likely encounter it frequently when dealing with raw data, which is almost always messy.


```python
print(frame3)
```

In this example, both Nevada and Texas have missing data. We can use the `isna` method (or `isnull` which is the older name for the same) to easily identify missing values.


```python
print(frame3.isna())
```

The `isna` operation is applied to every value in the DataFrame, resulting in an array of Boolean values. This is called a *Boolean mask* because, like a physical mask, it covers some things (`False`) while leaving others exposed (`True`). Later, we will learn how to use these masks to select only the values we want to operate on.

The `notna` method returns the opposite result.


```python
print(frame3.notna())
```

### Counting Missing Values

In Python, `True` and `False` are alternative representations of `1` and `0`, respectively. We can take advantage of this quirk to easily calculate the number of missing values in a Boolean mask using the `sum` method.


```python
frame3.isna().sum()
```

Connecting a sequence of operations in this manner is called *method chaining* and it works any time that the output of one method is a suitable input for the next.

In this example we run `isna` on the data, creating an array of Boolean values. That is passed to the `sum` method of `pd.DataFrame`, which returns the number of `True` values, each equivalent to `1`.

Pandas commonly uses method chaining to build end to end data processing pipelines. Within reason, method chaining can help make code more readable by eliminating the need for intermediate variables. Here is a non-running example to get a feel for where we are headed:

```python
result = (
  df.dropna()
    .groupby('category')
    .agg({'value': 'mean'})
    .sort_values('value')
    .reset_index()
)
```

Note how the outer parentheses are used to allow the creative use of whitespace, making the code very readable.

### NaN Propagation

It is important to note that the presence of missing values affects computations. Any operation involving `NaN` will produce a `NaN` result, regardless of any other operands.


```python
1 + np.nan
```

NaN is said to *propagate* through calculations, spreading to all results derived from it. This ensures that missing or invalid data is *exposed, not supressed*.

You may also see missing values represented by `None` and/or `pd.NA`. The latter was introduced as an alternative to `NaN` that works more consistently across all data types. We'll discuss this more as required.

### Handling Missing Values

In order to get useful results, missing values (however represented) must be dealt with. There are basically four alternatives:

1. Correct the source of the data
2. Drop missing values
3. Replace missing values
4. Flag them in another way and work around them

The approach to use, methods available, which to use, and how to implement them, is a topic for future study.

## Operating on Data in Pandas

Pandas builds on NumPy's strengths for quick element-wise operations by preserving the context of Series and DataFrame objects.

### Unary Operations Preserve Indicies

To begin with, any NumPy numerical function will work on a Series or DataFrame. For unary operations - those that modify an existing object - the index order is preserved.


```python
rng = np.random.default_rng(42)
ser = pd.Series(rng.integers(0, 10, 4))
ser
```


```python
df = pd.DataFrame(rng.integers(0, 10, (3, 4)),
                  columns=['A', 'B', 'C', 'D'])
print(df)
```


```python
# exponential function in numpy
np.exp(ser)
```

Note the indices are preserved. The same is true for any NumPy calculation on a DataFrame.


```python
print(np.sin(df * np.pi / 4))
```

### Binary Operations Align Indicies

For operations involving two Series or DataFrame objects, Pandas will maintain the alignment of indicies.


```python
area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
                  'California': 423967}, name='area')

population = pd.Series({'California': 39538223, 'Texas': 29145505,
                        'Florida': 21538187}, name='population')

print('viewed side by side:')
print(pd.DataFrame({"area": area, "pop": population}))
```

These Series are unaligned - they don't share the same set of row labels. Alaska is not included in the population data and Florida is missing from area.

What happens if we divide the two series objects to compute the population density?


```python
print(population / area)
```

We get all the rows from both (the **union** of row labels), where the density is `NaN` for any result where either operand was missing.

Pandas does the "dirty work" of ensuring that `population['Florida']` is divided by `area['Florida']` and not the row with the numerically equivalent index position, `area['California']`. Below is the side by side output above, modified to add an index column for both `area` (i), and `pop` (j).

```text
            area       i    pop       j
Alaska      1723337.0  0         NaN
California   423967.0  1  39538223.0  0
Florida           NaN     21538187.0  1
Texas        695662.0  2  29145505.0  2
```

What would NumPy do?


```python
# Pandas stores the ndarray in the values attribute of Series / DataFrame objects
a_np = area.values
p_np = population.values
print(type(a_np), type(p_np))
p_np / a_np
```

Only one of these results is correctly calculated - the one for Texas, which is the last element in both `area` and `population`, so is aligned by coincidence.

To do this in base Python you would need to explicitly handle looping through each row calculation (which NumPy does implicitly) while ensuring the rows are aligned (which only Pandas does).

The benefits of Pandas' automatic row alignment should be pretty obvious from this example.

Pandas supports a wide range of arithmetic, comparison, and other operations on Series and DataFrames. Each type is briefly introduced below. We will build on these operations as required.

### Arithmetic Operations

Typical arithmetic operations between scalars, Series, and DataFrames are performed in element-wise fashion.


```python
# Create sample sales data

df = pd.DataFrame({
    'product': ['Apple', 'Banana', 'Orange', 'Mango', 'Kiwi'],
    'price': [0.99, 0.59, 0.89, 2.99, 1.99],
    'quantity': [100, 120, 80, 45, 75],
    'height_inches': [2.5, 7.0, 3.0, 4.0, 2.0]
})

print(df)
```


```python
# Element-wise operations between Series or DataFrames
print("10% discount:\n")
print(df['price'] * 0.9)
```


```python
# Create new column with total
df['total'] = df['price'] * df['quantity']
print("With totals:\n")
print(df)
```


```python
# Convert measurements
df['height_cm'] = df['height_inches'] * 2.54
print("With heights in cm:\n")
print(df)
```

### Comparison Operations

Traditional comparison operators (e.g. greater than) are supported, as well as some methods provide an alternative interface to more complex comparisons (aka "convenience functions"). Both return an array of Boolean values, which can be used for masking.


```python
# simple comparison
print("Expensive items (>$1):\n")
print(df['price'] > 1)
```

Any combination of comparisons and element-wise Boolean operators (e.g., `&` or `|`) can be used to construct more complex expressions.


```python
print("Moderate quantity items (50-100 units):\n")
print((df['quantity'] >= 50) & (df['quantity'] <= 100))
```

Or use an equivalent method, when available, to make code more concise and readable.


```python
# between is equivalent to the comparison above
print("\nModerate quantity items (50-100 units):\n")
print(df['quantity'].between(50, 100))
```

### Operations with Methods

Finally, `pd.Series` and `pd.DataFrame` objects offer a wide variety of additional methods for working with data.


```python
# Common methods like round
print("Rounded prices:\n")
print(df['price'].round(0))
```


```python
# Enforce minimum values
print("Quantity floored at 50:\n")
print(df['quantity'].clip(lower=50))
```


```python
# Sort by price descending
print("Sorted by price (highest first):\n")
print(df.sort_values('price', ascending=False))
```

Pandas has many, many methods (functions) and properties (values or special accessor objects) associated with the DataFrame and Series objects:


```python
# Get all attributes from the class
all_attributes = dir(pd.DataFrame)

# Filter to just public methods
public_attributes = [m for m in all_attributes if not m.startswith('_')]
print(f"Total public attributes: {len(public_attributes)}")

# Check if attributes are methods using the class
methods_only = [m for m in public_attributes 
                if callable(getattr(pd.DataFrame, m))]

print(f"Methods: {len(methods_only)}")
print(f"Properties: {len(public_attributes) - len(methods_only)}")
```

Some major categories:

**Data Selection/Indexing:** `loc`, `iloc`, `at`, `iat`, `head`, `tail`, `sample`, `query`, `filter`

**Computation/Stats:** `sum`, `mean`, `median`, `std`, `var`, `min`, `max`, `cumsum`, `cumprod`, `describe`, `corr`, `cov`

**Data Manipulation:** `merge`, `join`, `concat`, `append`, `pivot`, `pivot_table`, `melt`, `stack`, `unstack`, `explode`

**Cleaning:** `dropna`, `fillna`, `drop_duplicates`, `replace`, `interpolate`, `clip`

**Transformation:** `apply`, `applymap`, `map`, `transform`, `agg`, `aggregate`, `groupby`, `rolling`, `expanding`, `ewm`

**Type Conversion:** `astype`, `to_numpy`, `to_dict`, `to_json`, `to_csv`, `to_excel`, `to_sql`, `to_pickle`

**Reshaping:** `transpose`, `T`, `sort_values`, `sort_index`, `nlargest`, `nsmallest`, `rank`

**Info/Inspection:** `info`, `shape`, `dtypes`, `columns`, `index`, `empty`, `size`, `ndim`, `memory_usage`

**Plotting:** `plot`, `plot.bar`, `plot.line`, `plot.scatter`, `plot.hist`, `plot.box`, etc.

You can find an interactive mindmap with (most of?) them here: https://app.xmind.com/share/ugVH30g4

Remember, `help` (e.g., `help(df.groupby)`) and TAB (autocomplete) are your friends!

### Pandas Methods Return new Objects

It is important to note that, *by default, **most*** Pandas Series and DataFrame methods return a new object of that type rather than performing in-place modification.

To capture the result of a method, you will need to assign it to a new variable, or reassign it to the original if you wish to replace it.


```python
# this does not modify df
df.sort_values('price')
```

Rarther than modifying the original `df`, this creates and *returns* the result. In Jupyter, the last value returned by a cell is echoed as output. Subsequent operations on `df` would be working with the original data.


```python
# this does update df with the sorted result
df = df.sort_values("price")
```

Here, there is no output to the screen because the returned value is reassigned to `df`. This illustrates an important difference between expressions and statements, and the implications in an interactive Python environment like Jupyter.

Pandas' preference for returned values over in-place modification is a departure from the typical base Python behavior, where mutable types like lists are modified in place. Why?

Method chaining.[1] As we discussed above, when working with data it is common to transform an input through a series of steps. This is well suited to method chaining, where the results of one operation are the input for the next. But this requires each method to *return* its result so that it can be handed off to the next method in the chain. In-place modification is incompatible with method chaining and therefore is not the default behavior in Pandas.

This deliberate design choice reflects a key insight about all software: the gap between how we think something should work and how it actually works often stems from not grasping the developers' mental model of proper usage. Rather than fighting against a tool's design, understanding what motivated those choices and leaning into them is how we get the most from any well-designed software

[1]: There are other reasons, but method chaining is the most impactful. In-place operations can lead to unintended changes to the original data, and relying on returned values eliminates that class of bugs. Also, some Pandas operations change the data's shape or index structure, making true in-place modification technically impossible or impractical.

Alternatively, many Pandas methods support the `inplace` argument, which automatically reassigns the result.


```python
# this has the same result as above
df.sort_values('price', inplace=True)
```

This is both longer and less explicit than simply reassigning the result to the original dataframe, but may be useful in some situations.

## Key Things to Remember

- Operations preserve index alignment
- When to use name-based column access vs `loc` vs `iloc` - be consistent and explicit!
- Missing values (NaN) propagate through operations
- Most operations are vectorized (no explicit loops needed)
- Methods generally return new objects rather than modifying in place

