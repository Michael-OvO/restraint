# 819: DS-1000 Task

## Prompt
Problem:

How do I convert data from a Scikit-learn Bunch object (from sklearn.datasets) to a Pandas DataFrame?

from sklearn.datasets import fetch_california_housing
import pandas as pd
data = fetch_california_housing()
print(type(data))
data1 = pd. # Is there a Pandas method to accomplish this?

A:

<code>
import numpy as np
from sklearn.datasets import fetch_california_housing
import pandas as pd
data = load_data()
</code>
data1 = ... # put solution in this variable
BEGIN SOLUTION
<code>

## What to do
- Edit `/solution/solution.py` so the code passes the DS-1000 tests.
- Do not access the internet or install new packages; required libraries are preinstalled in the Docker image.

## Notes
- Keep the variable names/signatures implied by the prompt/code_context.
- The evaluator uses the original DS-1000 `code_context` (`test_execution` / `test_string`).
