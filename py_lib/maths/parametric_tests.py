# https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
from scipy import stats


def student_t(data1, data2, p=0.05):
    """Student's t-test.
    Tests whether the means of two independent samples are significantly different
    Assumes IID behavior for both datasets
    Assumes normal distributions in each sample, and same variance in both

    Args:
        data1 (list or np.ndarray): Array of sample data
        data2 (list or np.ndarray): Array of sample data
        p (float, optional): P-value. Defaults to 0.05.
    """
    stat, pval = stats.ttest_ind(data1, data2)
    print(f'stat={stat:.3f}, p-value={pval:.3f}')
    if pval > p:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')


def paired_student_t(data1, data2, p=0.05):
    """Paired Student's t-test.
    Tests whether the means of two independent samples are significantly different
    Assumes IID behavior for both datasets
    Assumes normal distributions in each sample, and same variance in both
    Assumes observations across the datasets are paired

    Args:
        data1 (list or np.ndarray): Array of sample data
        data2 (list or np.ndarray): Array of sample data
        p (float, optional): P-value. Defaults to 0.05.
    """
    stat, pval = stats.ttest_rel(data1, data2)
    print(f'stat={stat:.3f}, p-value={pval:.3f}')
    if pval > p:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')


def analysis_variance(data, p=0.05):
    """Analysis of Variance test (ANOVA).
    Tests whether the means of two or more independent samples are significantly different
    Assumes IID behavior for both datasets
    Assumes normal distributions in each sample, and same variance in both

    Args:
        data (list): List of arrays
        p (float, optional): P-value. Defaults to 0.05.
    """
    stat, pval = stats.ttest_rel(*data)
    print(f'stat={stat:.3f}, p-value={pval:.3f}')
    if pval > p:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')


def parametric_test(*data):
    """Tests whether datasets have significantly different means.
    Parametric statistical methods often mean those methods that assume the data samples have a Gaussian distribution.

    Args:
        data (list): list of array sample data. If longer than 2, only ANOVA is done
    """
    print_len = 60

    if len(data) == 2:
        print("\n" + " Student's t-test ".center(print_len, '*'))
        student_t(*data)

        print("\n" + " Paired student's t-test ".center(print_len, '*'))
        paired_student_t(*data)

    print("\n" + " Analysis of Variance test ".center(print_len, '*'))
    analysis_variance(data)
