import numpy as np
from normality_tests import normality_test
from correlation_tests import correlation_test, chi_squared
from parametric_tests import parametric_test
from nonparametric_tests import nonparametric_test


data1 = np.random.normal(10, 2, 1000)
data1.sort()
data2 = np.random.normal(10, 2, 1000)
data2.sort()

_, cont1 = np.unique(data1.astype(int), return_counts=True)
_, cont2 = np.unique(data2.astype(int), return_counts=True)

print("\nNormality tests ...")
normality_test(data1)

print("\nCorrelation tests ...")
correlation_test(data1, data2)
chi_squared(cont1[:cont2.size], cont2[:cont1.size])

print("\nParametric tests ...")
parametric_test(data1, data2)

print("\nNonparametric tests ...")
nonparametric_test(data1, data2)
nonparametric_test(data1, data2, data2)
