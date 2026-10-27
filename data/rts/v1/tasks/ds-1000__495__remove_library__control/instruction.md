# 495: DS-1000 Task

## Prompt
Problem:
I have a file with arrays or different shapes. I want to zeropad all the array to match the largest shape. The largest shape is (93,13).
To test this I have the following code:
a = np.ones((41,13))
how can I zero pad this array to match the shape of (93,13)? And ultimately, how can I do it for thousands of rows? Specifically, I want to pad to the right and bottom of original array in 2D.
A:
<code>
import numpy as np
a = np.ones((41, 13))
shape = (93, 13)
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>

## What to do
- Edit `/solution/solution.py` so the code passes the DS-1000 tests.
- Do not access the internet or install new packages; required libraries are preinstalled in the Docker image.

## Notes
- Keep the variable names/signatures implied by the prompt/code_context.
- The evaluator uses the original DS-1000 `code_context` (`test_execution` / `test_string`).
