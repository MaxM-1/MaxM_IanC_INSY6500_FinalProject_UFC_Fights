# Intermediate Python

A brief discussion of the following topics:

- dictionaries and iterating
- functions: arguments / parameters and variable scope
- lambdas
- error handling

All of these are particularly relevant for our next topic, Pandas.

## Dictionaries Revisited

### Creation and Access


```python
scores = {'Alice': 85, 'Bob': 92, 'Charlie': 98}
```


```python
# key must exist to be accessed without error
scores['Alice']
```


```python
scores['Dan']
```

A common pattern for creating / updating keys uses `if ... else`:


```python
# adjust score if record exists, otherwise create record
if 'Dan' not in scores:
    scores['Dan'] = 0    # create new kv pair
else:
    scores['Dan'] += 10  # increase existing value

print(scores['Dan'])
```

The `get` method provides a safe way to access values without risking KeyError exceptions. It takes two arguments, the key and a default value if the key does not exist.


```python
# Same operation, much cleaner!
scores['Eddie'] = scores.get('Eddie', 0) + 10   # Creates with 10
scores['Alice'] = scores.get('Alice', 0) + 10   # Updates to 95

print(scores)
```

Another example of how `get` is commonly used.


```python
word_count = {}
text = "the cat and the dog and the bird"

# Without get(): requires if/else
for word in text.split():
    if word in word_count:
        word_count[word] += 1
    else:
        word_count[word] = 1

# With get(): single line per word
word_count = {}
for word in text.split():
    word_count[word] = word_count.get(word, 0) + 1
```

### Iteration

Nested dictionaries are commonly used.


```python
# Sample data dictionary (similar to what you'll see in data analysis)
student_scores = {
    'Alice': {'math': 92, 'science': 88, 'english': 95},
    'Bob': {'math': 78, 'science': 92, 'english': 81},
    'Charlie': {'math': 95, 'science': 79, 'english': 88}
}
```

Dictionaries can be iterated over by key, value, or item (ke-value pair):


```python
# Iterating through keys
for student in student_scores.keys():
    print(f"{student}: {student_scores[student]}")
```


```python
# Keys is the default, the method isn't required
for student in student_scores.keys():
    print(f"{student}: {student_scores[student]}")
```


```python
# Iterating through values only
all_math_scores = []
for scores in student_scores.values():
    all_math_scores.append(scores['math'])

print(all_math_scores)
```


```python
# Iterating through items (kv pairs)
for student, scores in student_scores.items():
    avg_score = sum(scores.values()) / len(scores)
    print(f"{student}'s average: {avg_score:.1f}")
```

### Comprehensions

Dictionaries also support comprehensions, using equivalent syntax:

```python
{k: v for element in iterable}
```

Simple examples:


```python
# Basic dictionary comprehension
numbers = [1, 2, 3, 4, 5]
squares = {n: n**2 for n in numbers}
print(squares)
```


```python
# From existing dictionary
scores = {'Alice': 85, 'Bob': 92, 'Charlie': 78}
curved = {name: score + 5 for name, score in scores.items()}
print(curved)
```


```python
# With filtering
passing = {name: score for name, score in scores.items() if score >= 80}
print(passing)
```

This pattern is commonly used to create new dicts from calculated and/or filtered values.


```python
# Create a new dictionary with calculated values
averages = {
    student: sum(scores.values()) / len(scores)
    for student, scores in student_scores.items()
}

print(averages)
```


```python
# Filter dictionary based on condition
high_performers = {
    student: scores 
    for student, scores in student_scores.items()
    if sum(scores.values()) / len(scores) > 85
}

print(high_performers)
```

## Functions: Arguments, Parameters, and Variable Scope

### Arguments and Parameters

Remember this distinction:

- argument refers to the items in a function **call**
- parameter refers to the items in a function **definition**

When a function is called, each argument can be specified either by position or keyword. It is up to the user.

Positional arguments must be provided in the exact order defined in the function. All positional arguments must be provided unless a default is defined in the function definition. Parameters with defaults need not be specified in the function call.


```python
# parameters defined with function
def greet(name, age, city='Auburn'):
    return f"{name}, {age}, from {city}"

# arguments passed by position here - order matters
greet("Alice", 25, "NYC")
```


```python
greet("Bob", 40)
```


```python
# order matters!
greet(55, 'Charlie', 'Tuskaloosa')
```

Keyword arguments are specified by name, order does not matter.


```python
greet(age=25, city="NYC", name="Alice")  # Any order
```

If positional and keyword arguments are mixed, the positional arguments must come first.


```python
greet("Alice", city="NYC", age=25)  # Mixed, positional first
```

Parameters with defaults must come after parameters without them. **Defaults are evaluated once, at the time the function is defined.** There are important implications of this, as we will demonstrate later!

### `args` and `kwargs`

The syntax `*args` and `**kwargs`, when used in a function definition, collects all the positional (`args`) and keyword (`kwargs`) in a tuple or dictionary, respectively. The names `args` and `kwargs` are just variables and can be anything, but are commonly used in Python. In the following example, `scores` and `options` are used instead.


```python
def analyze_scores(student_name, *scores, **options):
    """
    Demonstrates different parameter types
    - student_name: positional argument (required)
    - *scores: variable positional arguments (tuple)
    - **options: keyword arguments (dictionary)
    """
    verbose = options.get('verbose', False)
    include_median = options.get('include_median', True)
    
    if verbose:
        print(f"Analyzing {student_name}'s scores: {scores}")
    
    result = {
        'mean': sum(scores) / len(scores) if scores else 0,
        'max': max(scores) if scores else 0,
        'min': min(scores) if scores else 0
    }
    
    if include_median:
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        result['median'] = (sorted_scores[n//2] if n % 2 
                           else (sorted_scores[n//2-1] + sorted_scores[n//2])/2)
    
    return result

# Various ways to call the function
result1 = analyze_scores('Alice', 92, 88, 95)
result2 = analyze_scores('Bob', 78, 92, 81, verbose=True)
result3 = analyze_scores('Charlie', 95, 79, 88, verbose=True, include_median=False)
```

### Default Mutable Arguments (Common Pitfall)

As mentioned above, default arguments are only evaluated once, at the time a function is defined. This can lead to unexpected behavior if a mutable object is used as default.


```python
# WRONG - mutable default argument
def add_scores_wrong(*scores, scores_list=[]):
    scores_list.extend(list(scores))
    return scores_list
```

This approach creates the variable `scores_list` local to this function (more on this later) and assigns an empty list to it.

All future calls to that function will reference this original definition.

At first, this may seem to work as expected...


```python
list1a = []
list1a = add_scores_wrong(10, 20, 30)
print(list1a)
```


```python
list2a = add_scores_wrong(40, 50, 60)
print(list2a)
```

The original list persists! In fact, `list1` and `list2` are the same list:


```python
print(list1a)
print(list1a is list2a)
```

The correct pattern here is to set the default value to `None` and, on use, initialize the mutable value.


```python
# CORRECT - use None as default
def add_scores_correct(*scores, scores_list=None):
    if scores_list is None:
        scores_list = []
    scores_list.extend(list(scores))
    return scores_list
```


```python
list1b = []
list1b = add_scores_correct(70, 80, 90)
print(list1b)
```


```python
list2b = add_scores_correct(100, 110, 120)
print(list2b)
```

### Variable Scope and the LEGB Rule

Python resolves variables using LEGB order: Local → Enclosing → Global → Built-in

- Built-in = Python internals, below where we work
- **Global = base (aka module) level, where we start**
- Enclosing = ignore for this course, mostly
- **Local = within functions**

We will mostly ignore enclosing scope (more advanced), though it will come into play when (if?) we talk about decorators.

Values are **inherited down** this chain (BGEL) or **passed up** this chain (LEGB). Not the opposite.

For example, at the Global scope we have access to all Python built-ins (and can even overwrite them!), but those functions do not use Global values unless passed in as arguments.

Here's an example of Local and Global interaction...


```python
global_threshold = 80                 # Global scope

def grade(score):
    curve = 10                        # Local scope
    adjusted = score + curve          # also Local
    
    if adjusted >= global_threshold:  # 'global_threshold' inherited from global scope
        return 'Pass'
    return 'Fail'

grade(75)
```

In this example, `global_threshold` is available inside the function (Global → Local), but `curve` is **only** available *within the function*, not in the Global scope...


```python
print(curve)
```

This is another advantage of functions. They help keep the Global "namespace" clean, avoiding variable naming conflicts that can lead to bugs.

#### Common Gotcha: `UnboundLocalError`

When an assignment operation (e.g., `x += 1`) is encountered within a function, the varible is marked as local when the function is defined, *even if it was defined globally*.


```python
count = 0

def increment_count():
    print(f"Current count: {count}")  # Tries to read 'count'
    count += 1  # But this line makes Python treat 'count' as LOCAL
    
increment_count()
```

In this example, when `increment_count` is defined, the assignment in line 5 causes `count` to be flagged as a local variable, despite the fact that it is defined as a global on line 1. As a result there are two variables named `count`, one global, one local. When `increment_count` is called, line 4 references `count`, but it has no value associated. This results in the cryptic `UnboundLocalError` message.

The best way to avoid this is not to rely on inheriting from the global scope. Instead pass arguments in and return values, as shown in the version below.


```python
count = 0

def increment_count_v2(current_count):
    print(f"Current count: {current_count}")
    return current_count + 1

count = increment_count_v2(count)
```

Python does allow you to override LEGB with the `global` and `nonlocal` keywords, but this is discouraged, especially in this class.

The language of this section is less precise than reality, but it is sufficient for our needs.

## Lambda Functions

Lambdas are anonymous functions perfect for simple operations:


```python
# Traditional function
def square(x):
    return x ** 2

# Equivalent lambda
square_lambda = lambda x: x ** 2

# Multiple parameters
multiply = lambda x, y: x * y

# Conditional logic in lambda
grade_pass = lambda score: 'Pass' if score >= 60 else 'Fail'
```

They are particularly useful with built-in functions and methods like `sorted()` and `.sort()`:


```python
scores = [78, 92, 45, 88, 67, 95, 81]

# Map: Transform each element
curved_scores = list(map(lambda x: min(x + 10, 100), scores))

# Filter: Select elements meeting criteria
passing_scores = list(filter(lambda x: x >= 70, scores))

# Sorted: Custom sorting
students = [
    {'name': 'Alice', 'score': 92},
    {'name': 'Bob', 'score': 78},
    {'name': 'Charlie', 'score': 95}
]

# Sort by score
by_score = sorted(students, key=lambda s: s['score'], reverse=True)

# Sort by name length, then alphabetically
by_name_complexity = sorted(students, 
                          key=lambda s: (len(s['name']), s['name']))
```

## Error Handling

### Basic Try-Except Structure


```python
def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Warning: Division by zero attempted")
        return None
    except TypeError as e:
        print(f"Type error: {e}")
        return None
```

### Multiple Exception Handling


```python
def process_student_data(data):
    try:
        name = data['name']
        scores = data['scores']
        average = sum(scores) / len(scores)
        return {'name': name, 'average': average}
    
    except KeyError as e:
        print(f"Missing required field: {e}")
        return None
    
    except (TypeError, ValueError) as e:
        print(f"Invalid data format: {e}")
        return None
    
    except ZeroDivisionError:
        print("No scores provided")
        return {'name': data.get('name', 'Unknown'), 'average': 0}
    
    except Exception as e:
        # Catch-all for unexpected errors
        print(f"Unexpected error: {e}")
        return None
```

### Try-Except-Else-Finally Pattern


```python
def read_score_file(filename):
    file = None
    try:
        file = open(filename, 'r')
        scores = [float(line.strip()) for line in file]
    
    except FileNotFoundError:
        print(f"File {filename} not found")
        return []
    
    except ValueError as e:
        print(f"Invalid score format in file: {e}")
        return []
    
    else:
        # Executes only if no exception occurred
        print(f"Successfully read {len(scores)} scores")
        return scores
    
    finally:
        # Always executes, even if return happened
        if file:
            file.close()
            print("File closed")
```


```python

```
