# 869: DS-1000 Task

## Prompt
Problem:

How can I perform regression in sklearn, using SVM and a gaussian kernel?
Note to use default arguments. Thanks.

A:

<code>
import numpy as np
import pandas as pd
import sklearn
X, y = load_data()
assert type(X) == np.ndarray
assert type(y) == np.ndarray
# fit, then predict X
</code>
predict = ... # put solution in this variable
BEGIN SOLUTION
<code>

## What to do
- Edit `/solution/solution.py` so the code passes the DS-1000 tests.
- Do not access the internet or install new packages; required libraries are preinstalled in the Docker image.

## Notes
- Keep the variable names/signatures implied by the prompt/code_context.
- The evaluator uses the original DS-1000 `code_context` (`test_execution` / `test_string`).
