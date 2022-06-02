# stack
#====================
## 함수

## 전역 변수
size = 5 # 스택 공간 개수
stack = [None for _ in range(size)] # 변수를 담을 공간
top = -1 # 맨 위의 데이터의 위치를 잡는 변수

## 메인 기능
# push
top += 1
stack[top] = '커피'

top += 1
stack[top] = '녹차'

top += 1
stack[top] = '꿀물'

# pop
data = stack[top]
stack[top] = None
top -= 1
print(data)
