def manhattan(a, b):
    row_distance = abs(a[0] - b[0])
    col_distance = abs(a[1] - b[1])
    return row_distance + col_distance