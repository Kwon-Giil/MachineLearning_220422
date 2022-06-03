# 1 ~ num까지의 합을 구하는 함수
def addNum(num):
    if num <= 1: # 0번째 숫자를 1로 정의
        return 1
    return num + addNum(num-1)

print(addNum(10))