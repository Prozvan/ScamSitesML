
def binary_transform(result):
    arr = []
    for i in result:
        if i[0] >= 0.5: arr.append(1)
        else: arr.append(0)


    return arr