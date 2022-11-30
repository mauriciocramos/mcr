import sys
import numpy


def edit_distance(r, h):
    """
    From: https://github.com/zszyellow/WER-in-python
    
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.

    Main algorithm used is dynamic programming.

    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    """
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d


def get_step_list(r, h, d):
    """
    This function is to get the list of steps in the process of dynamic programming.

    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calculating the editing distance of h and r.
    """
    x = len(r)
    y = len(h)
    lst = []
    while True:
        if x == 0 and y == 0: 
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1] and r[x-1] == h[y-1]: 
            lst.append("e")
            x = x - 1
            y = y - 1
        elif y >= 1 and d[x][y] == d[x][y-1]+1:
            lst.append("i")
            x = x
            y = y - 1
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1]+1:
            lst.append("s")
            x = x - 1
            y = y - 1
        else:
            lst.append("d")
            x = x - 1
            y = y
    return lst[::-1]


def aligned_print(lst, r, h, result):
    """
    This function is to print the result of comparing reference and hypothesis sentences in an aligned way.
    
    Attributes:
        lst    -> the list of steps.
        r      -> the list of words produced by splitting reference sentence.
        h      -> the list of words produced by splitting hypothesis sentence.
        result -> the rate calculated based on edit distance.
    """
    print("REF:", end=" ")
    for i in range(len(lst)):
        if lst[i] == "i":
            count = 0
            for j in range(i):
                if lst[j] == "d":
                    count += 1
            index = i - count
            print(" "*(len(h[index])), end=" ")
        elif lst[i] == "s":
            count1 = 0
            for j in range(i):
                if lst[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if lst[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) < len(h[index2]):
                print(r[index1] + " " * (len(h[index2])-len(r[index1])), end=" ")
            else:
                print(r[index1], end=" "),
        else:
            count = 0
            for j in range(i):
                if lst[j] == "i":
                    count += 1
            index = i - count
            print(r[index], end=" "),
    print("\nHYP:", end=" ")
    for i in range(len(lst)):
        if lst[i] == "d":
            count = 0
            for j in range(i):
                if lst[j] == "i":
                    count += 1
            index = i - count
            print(" " * (len(r[index])), end=" ")
        elif lst[i] == "s":
            count1 = 0
            for j in range(i):
                if lst[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if lst[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
                print(h[index2] + " " * (len(r[index1])-len(h[index2])), end=" ")
            else:
                print(h[index2], end=" ")
        else:
            count = 0
            for j in range(i):
                if lst[j] == "d":
                    count += 1
            index = i - count
            print(h[index], end=" ")
    print("\nEVA:", end=" ")
    for i in range(len(lst)):
        if lst[i] == "d":
            count = 0
            for j in range(i):
                if lst[j] == "i":
                    count += 1
            index = i - count
            print("D" + " " * (len(r[index])-1), end=" ")
        elif lst[i] == "i":
            count = 0
            for j in range(i):
                if lst[j] == "d":
                    count += 1
            index = i - count
            print("I" + " " * (len(h[index])-1), end=" ")
        elif lst[i] == "s":
            count1 = 0
            for j in range(i):
                if lst[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if lst[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
                print("S" + " " * (len(r[index1])-1), end=" ")
            else:
                print("S" + " " * (len(h[index2])-1), end=" ")
        else:
            count = 0
            for j in range(i):
                if lst[j] == "i":
                    count += 1
            index = i - count
            print(" " * (len(r[index])), end=" ")
    print("\nWER:", result)


def wer(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split()) 
    """
    # build the matrix
    d = edit_distance(r, h)

    # find out the manipulation steps
#     lst = getStepList(r, h, d)

    # mauricio's hack: avoid printing and return raw data
    return float(d[len(r)][len(h)]) / len(r)

#     # print the result in aligned way
#     result = float(d[len(r)][len(h)]) / len(r) * 100
#     result = str("%.2f" % result) + "%"
#     alignedPrint(lst, r, h, result)


if __name__ == '__main__':
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    with open(filename1, 'r', encoding="utf8") as ref:
        reference = ref.read().split()
    with open(filename2, 'r', encoding="utf8") as hyp:
        hypothesis = hyp.read().split()
    wer(reference, hypothesis)
