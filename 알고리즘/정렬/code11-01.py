import random
## 함수
def findMinIndex(ary) : # ary에서 제일 작은 위치를 찾기
    minIndex = 0
    for i in range(1, len(ary)) :
        if ( ary[minIndex] > ary[i]) :
            minIndex = i
    return minIndex

## 전역
before = [random.randint(1, 100) for _ in range(8)]
after = []

## 메인
print('정렬 전 -->', before)
for i in range(len(before)) :
    minPos = findMinIndex(before)
    after.append(before[minPos])
    del(before[minPos])
print('정렬 후 -->', after)