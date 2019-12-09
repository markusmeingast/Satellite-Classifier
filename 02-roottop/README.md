Default U-Net model with varying starting filter sizes:

Power | Parameters | Train  | Test   | h5 Size | TPU
------|-----------:|--------|------- | -------:| ---
1     | 30.000     | 92.50% | 92.64% | 1MB   | Works
2     | 122.000    | 97.20% | 97.24% | 2MB   | Works
3     | 486.000    | 98.32% | 98.21% | 6MB   | Works
4     | 1.940.000  | 98.16% | 98.02% | 22MB    | Works
5     | 7.760.000  | 98.50% | 98.18% | 89MB    | alloc. error
6     | 31.000.000 | 99.18% | 98.78% | 312MB   | alloc. error

U-Net with one less layer shortcut

Power | Parameters | Train | Test | h5 Size | TPU
------|-------:|------|---- | --- | ---
3 | 121.000 | 97.92% | 97.83% |  |
