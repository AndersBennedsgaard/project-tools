from scipy import stats


def mann_whitney(data1, data2, p=0.05):
    """Mann-Whitney U test.
    Tests whether the distributions of two independent samples are equal or not.
    Assumes IID behavior for both datasets
    Assumes observations in each sample can be ranked

    Args:
        data1 (list or np.ndarray): Array of sample data
        data2 (list or np.ndarray): Array of sample data
        p (float, optional): P-value. Defaults to 0.05.
    """
    stat, pval = stats.mannwhitneyu(data1, data2)
    print(f'stat={stat:.3f}, p-value={pval:.3f}')
    if pval > p:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')


def kruskal_wallis(data1, data2, p=0.05):
    """Kruskal-Wallis H Test.
    Tests whether the distributions of two or more independent samples are equal or not.
    Assumes IID behavior for both datasets.
    Assumes observations in each sample can be ranked.

    Args:
        data1 (list or np.ndarray): Array of sample data
        data2 (list or np.ndarray): Array of sample data
        p (float, optional): P-value. Defaults to 0.05.
    """
    stat, pval = stats.kruskal(data1, data2)
    print(f'stat={stat:.3f}, p-value={pval:.3f}')
    if pval > p:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')


def wilcoxon_signed_rank(data1, data2, p=0.05):
    """Wilcoxon Signed-Rank Test.
    Tests whether the distributions of two paired samples are equal or not.
    Assumes IID behavior for both datasets
    Assumes observations in each sample can be ranked
    Assumes observations across each sample are paired.

    Args:
        data1 (list or np.ndarray): Array of sample data
        data2 (list or np.ndarray): Array of sample data
        p (float, optional): P-value. Defaults to 0.05.
    """
    stat, pval = stats.wilcoxon(data1, data2)
    print(f'stat={stat:.3f}, p-value={pval:.3f}')
    if pval > p:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')


def friedman(data, p=0.05):
    """Friedman Test.
    Tests whether the distributions of two or more paired samples are equal or not.
    Assumes IID behavior for both datasets.
    Assumes observations in each sample can be ranked.
    Assumes observations across each sample are paired.

    Args:
        data (list): List of arrays
        p (float, optional): P-value. Defaults to 0.05.
    """
    stat, pval = stats.friedmanchisquare(*data)
    print(f'stat={stat:.3f}, p-value={pval:.3f}')
    if pval > p:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')


def nonparametric_test(*data):
    """Tests whether datasets have significantly different distributions.
    Nonparametric statistics are those methods that do not assume a specific distribution to the data.

    Args:
        data (list): list of array sample data. If longer than 2, only the Friedman test is done
    """
    print_len = 60

    if len(data) == 2:
        print("\n" + " Mann-Whitney U test ".center(print_len, '*'))
        mann_whitney(*data)

        print("\n" + " Kruskal-Wallis H Test ".center(print_len, '*'))
        kruskal_wallis(*data)

        print("\n" + " Wilcoxon Signed-Rank Test ".center(print_len, '*'))
        wilcoxon_signed_rank(*data)

    elif len(data) > 2:
        print("\n" + " Friedman Test ".center(print_len, '*'))
        friedman(data)
    else:
        print("Number of datasets is not 2 or above. Skipping ...\n")
