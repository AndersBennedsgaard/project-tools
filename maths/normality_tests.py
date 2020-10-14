# https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
from scipy import stats


def shapiro_wilk(data, p=0.05):
    """Shapiro-Wilk test.
    Tests the null hypothesis that the data was drawn from a normal distribution.
    Assumes IID behavior

    Args:
        data (list or np.ndarray): Array of sample data
        p (float, optional): P-value. Defaults to 0.05.
    """
    stat, pval = stats.shapiro(data)
    print(f'stat={stat:.3f}, p-value={pval:.3f}')
    if pval > p:
        print('Probably Gaussian')
    else:
        print('Probably not Gaussian')


def d_agostino(data, p=0.05):
    """D'Agostino's K^2 test.
    Tests the null hypothesis that the data was drawn from a normal distribution.
    Assumes IID behavior

    Args:
        data (list or np.ndarray): Array of sample data
        p (float, optional): P-value. Defaults to 0.05.
    """
    stat, pval = stats.normaltest(data)
    print(f'stat={stat:.3f}, p-value={pval:.3f}')
    if pval > p:
        print('Probably Gaussian')
    else:
        print('Probably not Gaussian')


def anderson_darling(data):
    """Anderson-Darling test.
    Tests the null hypothesis that the data was drawn from a normal distribution.
    Assumes IID behavior

    Args:
        data (list or np.ndarray): Array of sample data
    """
    result = stats.anderson(data)
    print(f'stat={result.statistic:.3f}')
    for sl, cv in zip(result.significance_level, result.critical_values):
        if result.statistic < cv:
            print(f'Probably Gaussian at the {sl:.1f}% level')
        else:
            print(f'Probably not Gaussian at the {sl:.1f}% level')


def normality_test(data):
    """Tests the null hypothesis that the data was drawn from a normal distribution.
    Assumes IID behavior

    Args:
        data (list or np.ndarray): Array of sample data
    """
    print_len = 60

    print("\n" + " Shapiro-Wilk test ".center(print_len, '*'))
    shapiro_wilk(data)

    print("\n" + " D'Agostino K^2 test ".center(print_len, '*'))
    d_agostino(data)

    print("\n" + " Anderson-Darling test ".center(print_len, '*'))
    anderson_darling(data)
