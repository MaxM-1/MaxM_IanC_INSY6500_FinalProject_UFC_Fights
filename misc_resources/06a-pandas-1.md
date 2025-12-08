# Intro to Pandas

Pandas is designed primarily for manipulating tabular (i.e., 2D).

The *DataFrame* is Pandas' primary data structure - a 2D table where columns are *Series* (1D arrays) that can have different types. This matches how we typically think about data: observations (rows) described by named attributes (columns).


```python
import numpy as np
import pandas as pd  # similar convention for name
```


```python
table = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# 2D array from a nested list (list of lists, or table)
arr_2d = np.array(table)
print(arr_2d)
type(arr_2d)
```


```python
# Let's convert our table array into a pandas DataFrame
# Same data, but now with column names
df = pd.DataFrame(arr_2d, columns=['A', 'B', 'C'])
print(df)
```

Pandas simplifies column access by supporting named references.


```python
# NumPy uses somewhat cryptic indexing syntax
print("NumPy - first column:", arr_2d[:,0])

# Pandas allows named columns
print("\nPandas - column 'A':")
print(df['A'])  # direct access by name
```

Conceptually,

- A Pandas Series is like a 1D NumPy array with (optional) labels for indicies
- A Pandas Dataframe is like a group of named Series that represent the columnar data.

Where NumPy arrays are homogeneous, every column (Series) in a Pandas DataFrame has it's own type.

This is much more in line with real world use cases, where each attribute represents different observed values. For example, a table of collected data might include the age, sex / gender, country of origin, date of birth, and score for each participant in a study.


```python
# Create a DataFrame with example participant data
df = pd.DataFrame({
   'age': [25, 34, 28, 22],
   'gender': ['F', 'M', 'F', 'NB'],
   'country': ['USA', 'India', 'Canada', 'Mexico'],
   'birth_date': pd.to_datetime(['1999-03-15', '1990-08-22', '1996-11-03', '2002-05-30']),
   'score': [82.5, 91.0, 88.5, 79.0]
})

# Show the data types of each column
print("Column types:")
print(df.dtypes)

# Show the first few rows of data
print("\nFirst few rows:")
print(df)
```

Here we have constructed a table as a dictionary of lists, where:

- the dictionary represents the table
- keys are named columns
- values are the rows, the attribute value for that observation

In the example above we've directly converted a dictionary constructed in that fashion into a table (Pandas dataframe).

## Core Data Structures

Pandas is built on top of NumPy and adopts many of its idioms. But Pandas is designed for working with heterogenous tabular data, where NumPy is built for n-dimensional homogeneous numerical arrays.

Note: Parts of this notebook is adapted from chapter 5 of McKinney, which is available in HTML format here: [Pandas Basics](https://wesmckinney.com/book/pandas-basics).

Other portions draw from VanderPlas, chapters 13-16.

Our work with Pandas will rely on an understanding of its primary data structures, the `pd.Series` and `pd.DataFrame`.

### Pandas Series

`Series` wraps a one-dimensional NumPy array with additional functionality, including named indicies. By default, numbered indicies are assigned.

For example, imagine some results from a focus test conducted at various sites around the country.


```python
res = pd.Series([11.8, 30., 4.2, 3.4])
res
```

This output shows the default numeric indicies, corresponding values, and data type for the object.

The values and index can be accessed through attributes of the same name.


```python
res.values
```


```python
type(res.values)
```

The values are a standard `ndarray`, but the index is a special type.


```python
res.index
```

The index can consist of any value type. To specify the labels, use the index parameter.


```python
res_1 = pd.Series([11.8, 30., 4.2, 3.4],
                 index=["Orlando", "Auburn", "Atlanta", "Birmingham"])
res_1
```

This creates an association between the site and its results data. In base Python, associations of this type are typically represented by `dict` (dictionary) objects.

You can easily convert a Pandas Series into a Python dictionary with the `to_dict` method.


```python
res_1.to_dict()
```

Python dictionaries map keys (e.g. site names) to values (e.g. results). They are represented by a comma separated list of key:value pairs surrounded by curly brackets.

Given a dictionary, you can directly create a Series with labeled indicies.


```python
data = {"Miami": 10.2, "Auburn": 15.25, "Birmingham": 7.1, "Tuscaloosa": 1.0}
res_2 = pd.Series(data)
res_2
```

Elements of a Series can be accessed by label using the `[]` operator. Index based access in this fashion is ambiguous and discouraged. In fact, future versions of Pandas will not support index based access for `pd.Series` objects.


```python
print(res_1['Orlando'])  # clear intent
print(res_1[0])          # do we mean a label named `0` or index `0`?
```

Instead, it is better to use `loc` and `iloc` methods, as described in the subsequent lecture, for most Series and DataFrame access. This makes the interface consistent and explicit.

### Pandas DataFrame

A DataFrame represents tabular data. It contains an ordered, named collection of columns, each of which is a Series. Because of the associative relationship between names and columns, a DataFrame can be thought of as a dictionary of Series with a shared index.

As such, it is common to construct a DataFrame from a Python dictionary, where keys are the column names and values are equal-length lists.


```python
data = {"state": ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada"],
            "year": [2000, 2001, 2002, 2001, 2002, 2003],
            "pop": [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)
print(frame)
```

Again, the default numerical index is provided. Columns are listed in the order of keyes in the data.

You can specify the column order with the argument of the same name, which takes a list of names. If a new column name is included, missing values will result.


```python
frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'])
frame2
```

`NaN` (Not A Number) is commonly used to represent missing values in Pandas. We'll discuss it further in a later section.



Columns can be accessed by name or as attributes using dot notation.


```python
frame2['state']
```


```python
frame2.year
```

Note that named access works for any column name where dot notation will only work when the column name is a valid Python variable name that does not conflict with other methods. As a result, name based access is considered best practice.

To access more than one column by name, specify a list of their names:


```python
frame2[['state', 'pop']]
```

We can extend the comparison of a Series and a Dictionary to include row labels. As a collection of named columns can be represented by a dictionary of names and column data, each column can be thought of as a collection of named rows.


```python
# nested dictionary of col_name:col_data
# where col_data is a dict of year:value
populations = {
    "Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6},
    "Nevada": {2001: 2.4, 2002: 2.9},
    "Texas": {2000: 8.4, 2001: 8.8}
}

frame3 = pd.DataFrame(populations)
print(frame3)
```


```python

```
