# HW3B - Pandas Fundamentals

See Canvas for details on how to complete and submit this assignment.

## Introduction

This assignment transitions you from NumPy's numerical array operations to Pandas' powerful tabular data manipulation. While NumPy excels at homogeneous numerical arrays, Pandas is designed for the heterogeneous, labeled data that characterizes most real-world datasets—mixing dates, categories, numbers, and text within the same table.

You'll work with real bike share data from Chicago's Divvy system to answer questions about urban transportation patterns. Through three progressively complex problems—exploring usage patterns, analyzing rider behavior, and conducting temporal analysis—you'll discover why Pandas has become the standard tool for data analysis in Python.

The assignment emphasizes Pandas' design philosophy: named column access, explicit indexing methods (loc/iloc), handling missing data, and method chaining for readable data pipelines. You'll also see how Pandas builds on NumPy while adding the structure and convenience needed for practical data science work.

This assignment should take 3-5 hours to complete.

Before submitting, ensure your notebook:

- Runs completely with "Kernel → Restart & Run All"
- Includes thoughtful responses to all interpretation questions
- Uses clear variable names and follows good coding practices
- Shows your work (don't just print final answers)

### Learning Objectives

By completing this assignment, you will be able to:

1. **Construct and manipulate Pandas data structures**
   - Create DataFrames from dictionaries and CSV files
   - Distinguish between Series and DataFrame objects
   - Set and reset index structures appropriately
   - Understand when operations return views vs copies
2. **Apply explicit indexing paradigms**
   - Use `loc[]` for label-based data access
   - Use `iloc[]` for position-based data access
   - Access columns using bracket notation
   - Explain when each indexing method is appropriate
3. **Diagnose and explore datasets systematically**
   - Use `info()`, `describe()`, `head()`, and `dtypes` to understand data structure
   - Identify missing values with `isna()` and `notna()`
   - Calculate summary statistics across different axes
   - Interpret value distributions with `value_counts()`
4. **Filter data with boolean indexing and queries**
   - Combine multiple conditions with `&`, `|`, and `~` operators
   - Use `isin()` for membership testing
   - Apply `query()` for readable complex filters
   - Understand how index alignment affects operations
5. **Work with datetime data**
   - Parse dates during CSV loading
   - Extract temporal components with the `.dt` accessor
   - Filter data by date ranges
   - Create time-based derived features
6. **Connect Pandas patterns to data analysis workflows**
   - Formulate questions that data can answer
   - Choose appropriate methods for different analysis tasks
   - Interpret results in domain context
   - Recognize when vectorized operations outperform apply()

### Generative AI Allowance

You may use GenAI tools for brainstorming, explanations, and code sketches if you disclose it, understand it, and validate it. Your submission must represent your own work and you are solely responsible for its correctness.

### Scoring

Total of 90 points available, will be graded out of 80. Scores of >100% are allowed.

Distribution:

- Tasks: 48 pts
- Interpretation: 32 pts
- Reflection: 10 pts

Points by Problem:

- Problem 1: 3 tasks, 10 pts
- Problem 2: 4 tasks, 14 pts
- Problem 3: 4 tasks, 14 pts
- Problem 4: 3 tasks, 10 pts

Interpretation Questions:

- Problem 1: 3 questions, 8 pts
- Problem 2: 4 questions, 8 pts
- Problem 3: 3 questions, 8 pts
- Problem 4: 3 questions, 8 pts

Graduate differentiation: poor follow-up responses will result in up to a 5pt deduction for that problem.

## Dataset: Chicago Divvy Bike Share

The dataset you will analyze is based on real trip information from Divvy, Chicago's bike share system. It contains individual trips with start/end times, station information, and rider type.

Dataset homepage: https://divvybikes.com/system-data

Each trip includes:

- Trip start and end times (datetime)
- Start and end station names and IDs
- Rider type (member vs casual)
- Bike type (classic, electric, or docked)

Chicago's Department of Transportation uses this data to optimize station placement, understand usage patterns, and improve service. You'll explore similar questions that real transportation analysts investigate.

## Problems

### Problem 1: Creating DataFrames from Scratch

Before loading data from files, you need to understand how Pandas structures are built. In this problem, you'll create Series and DataFrames manually using Python's built-in data structures. This is a quick warmup to establish the fundamentals.

#### Task 1a: Create a Series

Create a Series called `temperatures` representing daily high temperatures for a week:

- Monday: 72°F
- Tuesday: 75°F  
- Wednesday: 68°F
- Thursday: 71°F
- Friday: 73°F

Use the day names as the index. Print the Series and its data type.

##### Your Code


```python
import pandas as pd

# Task 1a code here...
```

#### Task 1b: Create a DataFrame from a Dictionary

Create a DataFrame called `products` with the following data:

| product | price | quantity |
|---------|-------|----------|
| Widget  | 19.99 | 100 |
| Gadget  | 24.99 | 75 |
| Doohickey | 12.49 | 150 |

Use a dictionary where keys are column names and values are lists. Print the DataFrame and report its shape.

##### Your Code


```python
# Task 1b code here...

```

#### Task 1c: Access DataFrame Elements

Using the `products` DataFrame from Task 1b, extract and print:

1. The `price` column as a Series
2. The `product` and `quantity` columns as a DataFrame (using a list of column names)

##### Your Code


```python
# Task 1c code here...

```

#### Interpretation

Answer the following questions (briefly / concisely) in the markdown cell below:

1. Data structure mapping: When you create a DataFrame from a dictionary (like in Task 1b), what do the dictionary keys become? What do the values become?
2. Bracket notation: Why does `df['price']` return a Series, but `df[['price']]` return a DataFrame? What's the difference in what you're asking for?
3. Index purpose: In Task 1a, you used day names as the index instead of default numbers (0, 1, 2...). When would a custom index like this be more useful than the default numeric index?

##### Your Answers

*Problem 1 interpretation here*

### Problem 2: Loading and Initial Exploration

Before starting this problem, make sure you are working in a copy of this file in the `my_repo` folder you created in HW2a. You must also have a copy of the file `202410-divvy-tripdata-100k.csv` in a subdirectory called `data`. That file structure is illustrated below.

```text
~/insy6500/my_repo
└── homework
    ├── data
    │   └── 202410-divvy-tripdata-100k.csv
    └── hw3b.ipynb
```

#### Task 2a: Load and Understand Raw Data

Start by loading the data "as-is" to get a general understanding of the overall structure and how Pandas interprets it by default.

Note on file paths: The provided code uses `Path` from Python's `pathlib` module to handle file paths. Path objects work consistently across operating systems (Windows uses backslashes `\`, Mac/Linux use forward slashes `/`), automatically using the correct separator for your system. The provided code defines `csv_path` which should be used as the filename in your `pd.read_csv` to load the data file.

1. Use `pd.read_csv` to load `csv_path` (provided below) without specifying any other arguments. Assign it to the variable `df_raw`.
2. Use the methods we described in class to explore the shape, structure, types, etc. of the data. In particular, consider which columns represent dates or categories.
3. Note the amount of memory used by the dataset. See the section on memory diagnostics in notebook 07a for appropriate code snippets using `memory_usage`.

##### Your Code


```python
import pandas as pd
import numpy as np
from pathlib import Path

# create a OS-independent pointer to the csv file created by Setup
csv_path = Path('./data/202410-divvy-tripdata-100k.csv')

# load and explore the data below (create additional code / markdown cells as necessary)

```

#### Task 2b: Reload with Proper Data Types

1. Repeat step 2a.1 to reload the data. Use the `dtype` and `parse_dates` arguments to properly assign categorical and date types. Assign the result to the variable name `rides`.
2. After loading, use `rides.info()` to confirm the type changes.
3. Use `memory_usage` to compare the resulting size with that from step 2a.3.

##### Your Code


```python
# task 2b code here...

```

#### Task 2c: Explore Structure and Missing Data

Using the `rides` DataFrame from Task 2b:

1. Determine the range of starting dates in the dataframe using the `min` and `max` methods.
2. Count the number of missing values in each column. See the section of the same name in lecture 06b.
3. Convert the Series from step 2 to a DataFrame using `.to_frame(name='count')`, then add a column called 'percentage' that calculates the percentage of missing values for each column.

##### Your Code


```python
# task 2c code here...

```

#### Task 2d: Create Trip Duration Column and Set Index

Before setting the index, create a derived column for trip duration:

1. Calculate trip_duration_min by subtracting `started_at` from `ended_at`, then converting to minutes using `.dt.total_seconds() / 60`
3. Display basic statistics (min, max, mean) for the new column using `.describe()`
4. Show the first few rows with `started_at`, `ended_at`, and `trip_duration_min` to verify the calculation
5. Set `started_at` as the DataFrame's index. Verify the change by printing the index and displaying the first few rows.

##### Your Code


```python
# task 2d code here...

```

#### Interpretation

Reflect on problem 2 and answer (briefly / concisely) the following questions:

1. What types did Pandas assign to `started_at` and `member_casual` in Task 2a? Why might these defaults be problematic?
2. Look at the values in the station ID fields. Based on what you learned about git commit IDs in HW3a, how do you think the station IDs were derived?
3. Explain in your own words what method chaining is, what `df.isna().sum()` does and how it works.
4. Assume you found ~10% missing values in station columns but ~0% in coordinates. What might explain this? How might you handle the affected rows?

##### Your Answers

*Problem 2 interpretation here*

#### Follow-Up (Graduate Students Only)

Compare memory usage results in 2a.3 and 2b.3. What caused the change? Why are these numbers different from what is reported at the bottom of `df.info()`? Which should you use if data size is a concern?

Working with DataFrames typically requires 5-10x the dataset size in available RAM. On a system with 16GB, assuming about 30% overhead from the operating system and other programs, what range of dataset sizes would be safely manageable? Calculate using both 5x (optimistic) and 10x (conservative) multipliers, then explain which you'd recommend for reliable work.

##### Your Answers

*Problem 2 follow-up response here*

### Problem 3: Filtering and Transformation

With clean data loaded, you can now filter and transform it to answer specific questions. This problem focuses on Pandas' powerful indexing and filtering capabilities, along with creating derived columns that enable deeper analysis.

You'll continue working with the `rides` DataFrame from Problem 2, which has `started_at` set as the index.

#### Task 3a: Boolean Indexing and Membership Testing

Use boolean indexing and the `isin()` method to answer these questions:

1. How many trips were taken by *members* using *electric bikes*? Use `&` to combine conditions.
2. What percentage of all trips does this represent?
3. How many trips started at any of these three stations: "Streeter Dr & Grand Ave", "DuSable Lake Shore Dr & Monroe St", or "Kingsbury St & Kinzie St"? Use `isin()`.

Note: Remember to use parentheses around each condition when combining with `&`.

##### Your Code


```python
# Task 3a code here...

```

#### Task 3b: Create Derived Columns from Datetime

Add two categorical columns to the rides DataFrame based on trip start time:

1. `is_weekend`: Boolean column that is True for Saturday/Sunday trips. Use .dt.dayofweek on the index (Monday=0, Sunday=6).
2. `time_of_day`: String categories based on start hour:
   - "Morning Rush" if hour is 7, 8, or 9
   - "Evening Rush" if hour is 16, 17, or 18
   - "Midday" for all other hours

For step 2, initialize the column to "Midday", then use .loc[mask, 'time_of_day'] with boolean masks to assign rush hour categories. Extract hour using .dt.hour on the index.

After creating both columns, use value_counts() on time_of_day to show the distribution.

##### Your Code


```python
# Task 3b code here...

```

#### Task 3c: Complex Filtering with query()

Use the `query()` method to find trips that meet **all** of these criteria:
- Casual riders (not members)
- Weekend trips  
- Duration greater than 20 minutes
- Electric bikes

Report:
1. How many trips match these criteria?
2. What percentage of all trips do they represent?
3. What is the average duration of these trips?

Hint: Column names work directly in `query()` strings. Combine conditions with `and`.

##### Your Code


```python
# Task 3c code here...

```

#### Task 3d: Explicit Indexing Practice

Practice using `loc[]` and `iloc[]` for different selection tasks:

1. Use `iloc[]` to select the first 10 trips, showing only `member_casual`, `rideable_type`, and `trip_duration_min` columns
2. Use `loc[]` to select trips from October 15-17 (use date strings `'2024-10-15':'2024-10-17'`), showing the same three columns
3. Count how many trips occurred during this date range

Note: When using `iloc[]`, remember it's position-based (0-indexed). When using `loc[]` with the datetime index, you can slice using date strings.

##### Your Code


```python
# Task 3d code here...

```

#### Interpretation

Reflect on this problem and answer (briefly / concisely) the following questions:

1. `isin()` advantages: Compare using `isin(['A', 'B', 'C'])` versus `(col == 'A') | (col == 'B') | (col == 'C')`. Beyond readability, what practical advantage does `isin()` provide when filtering for many values (e.g., 20+ stations)?
2. Conditional assignment order: In Task 3b, why did we initialize all values to "Midday" before assigning rush hour categories? What would go wrong if you assigned categories in a different order, or didn't set a default?
3. `query()` vs boolean indexing: The `query()` method in Task 3c could have been written with boolean indexing instead. When would you choose `query()` over boolean indexing? When might boolean indexing be preferable despite being more verbose?

##### Your Answers

*Problem 3 interpretation here*


#### Follow-Up (Graduate Students Only)

Pandas supports a variety of indexing paradigms, including bracket notation (`df['col']`), label-based indexing (`loc[]`), and position-based indexing (`iloc[]`). The lecture recommended using bracket notation only for columns, and loc/iloc for everything else. Explain the rationale: why is this approach better than using bracket notation for everything, even though `df[0:5]` technically works for row slicing?

##### Your Answers

*Graduate follow-up interpretation here*

### Problem 4: Temporal Analysis and Export

Time-based patterns are crucial for understanding bike share usage. In this problem, you'll analyze when trips occur, how usage differs between rider types, and export filtered results. You'll use the datetime index you set in Problem 2 and the derived columns from Problems 2-3.

#### Task 4a: Identify Temporal Patterns

Use the datetime index to extract temporal components and identify usage patterns:

1. Extract the *hour* from the index and use `value_counts()` to find the most popular hour for trips. Report the peak hour and how many trips occurred during that hour.
2. Extract the *day name* from the index and use `value_counts()` to find the busiest day of the week. Report the day and number of trips.
3. Sort the results from step 2 to show days in order from Monday to Sunday (not by trip count). Use `sort_index()`.

Hint: Use `.dt.hour` and `.dt.day_name()` on the datetime index.

##### Your Code


```python
# Task 4a code here...

```

#### Task 4b: Compare Groups with groupby()

Use `groupby()` (introduced in 07a) to compare trip characteristics across different groups:

1. Calculate the average trip duration by rider type (`member_casual`). Which group takes longer trips on average?
2. Calculate the average trip duration by bike type (`rideable_type`). Which bike type has the longest average trip?
3. Count the number of trips by rider type using `groupby()` with `.size()`. Compare this with using `value_counts()` on the `member_casual` column - do they give the same result?

Note: Use single-key groupby only (one column at a time).

##### Your Code


```python
# Task 4b code here...

```

#### Task 4c: Filter, Sample, and Export

Create a filtered dataset for weekend electric bike trips and export it:

The provided code once again uses Path to create an `output` directory and constructs the full file path as `output/weekend_electric_trips.csv`. Use the `output_file` variable when calling `.to_csv()`.

1. Filter for trips where `is_weekend == True` and `rideable_type == 'electric_bike'`
2. Use `iloc[]` to select the first 1000 trips from this filtered dataset
3. Use `reset_index()` to convert the datetime index back to a column (so it's included in the export)
4. Export to CSV with filename `weekend_electric_trips.csv`, including only these columns: `started_at`, `ended_at`, `member_casual`, `trip_duration_min`, `time_of_day`
5. Use `index=False` to avoid writing the default numeric index to the file

After exporting, report how many total weekend electric bike trips existed before sampling to 1000.

##### Your Code


```python
# do not modify this setup code
from pathlib import Path

output_dir = Path('output')
output_dir.mkdir(exist_ok=True)
output_file = output_dir / 'weekend_electric_trips.csv'

# Task 4c code here...
# use the variable `output_file` as the filename for step 4

```

#### Interpretation

Reflect on this problem and answer the following questions:

1. `groupby() conceptual model`: Explain in your own words what `groupby()` does. Use the phrase "split-apply-combine" in your explanation and describe what happens at each stage.
2. `value_counts()` vs `groupby()`: In Task 4b.3, you compared two approaches for counting trips by rider type. When would you use `value_counts()` versus `groupby().size()`? Is there a situation where only one of them would work?
3. Index management for export: In Task 4c, why did we use `reset_index()` before exporting? What would happen if you exported with the datetime index still in place and used `index=False`?

##### Your Answers

*Problem 4 interpretation here*

#### Follow-Up (Graduate Students Only)

Compare `CSV` and _pickle_ formats for data storage and retrieval.

Pickle is Python's built-in serialization format that saves Python objects exactly as they exist in memory, preserving all data types, structures, and metadata. Unlike CSV (which converts everything to text), pickle is binary (not human readable) and maintains the complete state of your DataFrame. Also, pickle files only work in Python, while CSV is universal. Read more in the [Pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_pickle.html).

The code below investigates an interesting pattern: Do riders take longer trips from scenic lakefront stations even during rush hours? This could indicate tourists or recreational riders using these popular locations for leisure trips during typical commute times. The analysis filters for trips over 15 minutes that started from lakefront stations during morning (7-9am) or evening (4-6pm) rush hours, sorted by duration to see the longest trips first.

Run the code below, then answer the interpretation questions:


```python
import os

# the following lines were commented out since they were run in 4c
# from pathlib import Path
# output_dir = Path('output')

csv_file = output_dir / 'lakefront_rush_trips.csv'
pickle_file = output_dir / 'lakefront_rush_trips.pkl'

# Filter for interesting pattern: Long trips (>15 min) during rush hours 
# from lakefront stations, sorted by duration
lakefront_rush = (rides
    .loc[(rides.index.hour.isin([7, 8, 9, 16, 17, 18]))]
    .loc[(rides['start_station_name'].str.contains('Lake Shore|Lakefront', 
                                                    case=False, 
                                                    na=False))]
    .loc[rides['trip_duration_min'] > 15]
    .sort_values('trip_duration_min', ascending=False)
    .head(1000)
    .reset_index()
    [['started_at', 'ended_at', 'start_station_name', 'end_station_name',
      'member_casual', 'rideable_type', 'trip_duration_min']]
)

print(f"Found {len(lakefront_rush)} long rush-hour trips from lakefront stations")

# Export to both formats
lakefront_rush.to_csv(csv_file, index=False)
lakefront_rush.to_pickle(pickle_file)

# Compare file sizes
csv_size = os.path.getsize(csv_file) / 1024  # Convert to KB
pickle_size = os.path.getsize(pickle_file) / 1024
print(f"\nCSV file size: {csv_size:.2f} KB")
print(f"Pickle file size: {pickle_size:.2f} KB")
print(f"Size difference: {abs(csv_size - pickle_size):.2f} KB")

# Compare load times
print("\nLoad time comparison:")
print("CSV:")
%timeit pd.read_csv(csv_file)
print("\nPickle:")
%timeit pd.read_pickle(pickle_file)

# Check data type preservation
# Note: CSV load without parse_dates loses datetime types
csv_loaded = pd.read_csv(csv_file)
pickle_loaded = pd.read_pickle(pickle_file)

print("\nData types from CSV (without parse_dates):")
print(csv_loaded.dtypes)
print("\nData types from Pickle:")
print(pickle_loaded.dtypes)
```

After running the code, answer these questions:

1. Method chaining: The analysis uses method chaining with a specific formatting pattern:

   ```python
   result = (df
       .method1()
       .method2()
       .method3()
   )
   ```

   This wraps the entire chain in parentheses, allowing each method to appear on its own line without backslashes. Discuss why this makes formatting more readable, how it makes debugging easier, how it relates to seeing changes in the code with git diff, and what downsides heavy chaining might have.
3. Data types: Compare the dtypes from CSV versus pickle. What types were preserved by pickle that were lost in CSV? Why is this preservation significant for subsequent analysis?
4. Trade-offs: Given your observations about size, speed, and type preservation, when would you choose pickle over CSV for your work? When would CSV still be the better choice despite pickle's advantages?


*Graduate follow-up interpretation here*

## Reflection

Address the following questions in a markdown cell:

1. NumPy vs Pandas
   - What was the biggest conceptual shift moving from NumPy arrays to Pandas DataFrames?
   - Which Pandas concept was most challenging: indexing (loc/iloc), missing data, datetime operations, or method chaining? How did you work through it?
2. Real Data Experience
   - How did working with real CSV data (with missing values, datetime strings, etc.) differ from hw2b's synthetic NumPy arrays?
   - Based on this assignment, what makes Pandas well-suited for data analysis compared to pure NumPy?
3. Learning & Application
   - Which new skill from this assignment will be most useful for your own data work?
   - On a scale of 1-10, how prepared do you feel to use Pandas for your own projects? What would increase that score?
4. Feedback
   - Time spent: ___ hours (breakdown optional)
   - Most helpful part of the assignment: ___
   - One specific improvement suggestion: ___

### Your Answers

*Reflection here*
