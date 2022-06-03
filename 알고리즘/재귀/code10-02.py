## 함수
def openBox():
    global cnt
    print('상자 깡~!')
    cnt -= 1

    if cnt == 0:
        print('샤코 인형 넣기')
        return

    openBox()
    print('상자 닫기 ^^')

## 변수
cnt = 3

## 메인 기능
openBox()