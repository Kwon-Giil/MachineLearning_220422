import random
## 함수
def selctionSort(ary) :
    n = len(ary)
    for i in range(0, n-1) : # Cycle
        minIndex = i
        for k in range(i+1, n) :
            if (ary[minIndex] > ary[k]) :
                minIndex = k

        ary[i], ary[minIndex] = ary[minIndex], ary[i]

    return ary

## 변수
dataAry = [random.randint(1, 100) for _ in range(8)]

## 메인
print('정렬 전 -->', dataAry)
dataAry = selctionSort(dataAry) # sorted(배열)
print('정렬 후 -->', dataAry)