import random
## 함수
def binarySearch(ary, fData) :
    pos = -1
    start = 0
    end = len(ary) -1

    count = 0
    while (start <= end) :
        count += 1
        mid = (start + end) // 2
        if ary[mid] == fData :
            return mid, count
        elif fData > ary[mid] :
            start = mid + 1
        else :
            end = mid - 1

    return pos, count

## 전역
dataAry = [random.randint(1, 10000000) for _ in range(1000000)]
findData = random.choice(dataAry) # 랜덤 데이터 중에서 무작위 선택
dataAry.sort() # 데이터 정렬

## 메인
#print('배열-->', dataAry)
position, cnt = binarySearch(dataAry, findData)
if position == -1 :
    print(findData, '가 없음....')
else :
    print(findData, '가 ', position, ' 위치에 있음(', cnt , '회)')

