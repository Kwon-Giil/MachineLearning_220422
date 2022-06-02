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
if __name__ == '__main__':

    select = input('삽입(I), 추출(E), 확인(V), 종료(X)')

    while(select != 'x' and select != 'x'):
        if select == 'I' or select == 'i':
            data = input('입력할 데이터: ')
            push(data)
            print('스택 상태: ', stack)

        elif (select == 'E' or select == 'e'):
            data = pop()
            print('추출한 데이터:', data)
            print('스택 상태: ', stack)

        elif (select == 'V' or select == 'v'):
            data = peek()
            print('확인된 데이터: ', data)
            print('스택 상태: ', stack)

        else:
            print('입력이 잘못됨')

        select = input('삽입(I), 추출(E), 확인(V), 종료(X) 중 하나 선택')

    print('프로그램 종료')