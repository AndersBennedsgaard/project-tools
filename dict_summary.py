
SPACING = 0


def dict_summary(ar):
    """
    Giving a summary of a dictionary
    """
    global SPACING
    space = " " * SPACING
    print(space + str(type(ar)))
    if isinstance(ar, dict):
        for key in ar.keys():
            print(space + "Key: " + str(key))
            SPACING += 4
            dict_summary(ar[key])
            SPACING -= 4
    elif isinstance(ar, str):
        str_describe(ar)
    elif isinstance(ar, bool):
        bool_describe(ar)
    elif isinstance(ar, int):
        int_describe(ar)
    elif isinstance(ar, list):
        list_summary(ar)
    elif isinstance(ar, set):
        set_summary(ar)
    else:
        print(space + "Value: Unknown")


def str_describe(ar):
    global SPACING
    space = " " * SPACING
    print(space + "Value: " + ar)


def int_describe(ar):
    global SPACING
    space = " " * SPACING
    print(space + "Value: " + str(ar))


def bool_describe(ar):
    global SPACING
    space = " " * SPACING
    print(space + "Value: " + str(ar))


def list_summary(ar):
    global SPACING
    space = " " * SPACING
    print(space + "Number of entries: " + str(len(ar)))
    if len(ar) < 10:
        print(space + "Value: " + str(ar))
    else:
        print(space + "Value: " + str(ar[:10]) + " ... ")


def set_summary(ar):
    global SPACING
    space = " " * SPACING
    print(space + "Number of entries: " + str(len(ar)))
    if len(ar) < 10:
        print(space + "Value: " + str(ar))

