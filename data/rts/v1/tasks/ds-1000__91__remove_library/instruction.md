# 91: DS-1000 Task

## Prompt
Problem:
I have multi-index df as follows

                x  y
id  date            
abc 3/1/1994  100  7
    9/1/1994   90  8
    3/1/1995   80  9
Where dates are stored as str.

I want to parse date index. The following statement

df.index.levels[1] = pd.to_datetime(df.index.levels[1])
returns error:

TypeError: 'FrozenList' does not support mutable operations.

A:
<code>
import pandas as pd

index = pd.MultiIndex.from_tuples([('abc', '3/1/1994'), ('abc', '9/1/1994'), ('abc', '3/1/1995')],
                                 names=('id', 'date'))
df = pd.DataFrame({'x': [100, 90, 80], 'y':[7, 8, 9]}, index=index)
</code>
df = ... # put solution in this variable
BEGIN SOLUTION
<code>

## What to do
- Edit `/solution/solution.py` so the code passes the DS-1000 tests.
- Do not access the internet or install new packages; required libraries are preinstalled in the Docker image.

## Notes
- Keep the variable names/signatures implied by the prompt/code_context.
- The evaluator uses the original DS-1000 `code_context` (`test_execution` / `test_string`).
