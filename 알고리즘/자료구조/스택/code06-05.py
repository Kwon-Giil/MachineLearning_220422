# stack
#====================
## 함수
def isStackFull():
    global stack, size, top

    if (top >= size-1):
        return True

    else:
        return False

def isStackEmpty():
    global stack, size, top

    if(top <= -1):
        return True
    
    else:
        return False

def push(data):
    global stack, size, top

    if (isStackFull == True):
        print('스택이 꽉 찼습니다')
        return
    top += 1
    stack[top] = data

def pop() :
    global SIZE, stack, top
    if ( isStackEmpty() ) :
        print("스택이 비었습니다.")
        return None
    data = stack[top]
    stack[top] = None
    top -= 1
    return data

def peek() :
    global SIZE, stack, top
    if ( isStackEmpty() ) :
        print("스택이 비었습니다")
        return None
    return stack[top]


## 전역 변수
size = 5 # 스택 공간 개수
stack = [None for _ in range(size)] # 변수를 담을 공간
top = -1 # 맨 위의 데이터의 위치를 잡는 변수

# 메인 기능
push('커피')
push('녹차')
push('꿀물')
push('콜라')
push('환타')
# print(stack)

retData = pop()
print('추출-->', retData)
retData = pop()
print('추출-->', retData)
retData = pop()
print('추출-->', retData)
print(stack)

print('다음 나올 데이터-->', peek())