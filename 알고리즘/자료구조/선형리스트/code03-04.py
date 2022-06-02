## 함수 선언
def  add_data(friend) :
    katok.append(None)
    Klen = len(katok)
    katok[Klen-1] = friend

def  insert_data(position, friend) :
    katok.append(None)
    Klen = len(katok)

# 데이터를 뒤에 밀어내고 중간에 빈칸 생성
    for i in range(Klen-1, position, -1) :
        katok[i] = katok[i-1]
        katok[i-1] = None
        
    katok[position] = friend


def delete_data(position):
    Klen = len(katok) # katok배열의 길이를 저장할 변수 저장
    katok[position] = None

    for i in range(position+1, Klen, 1): # 삭제한 데이터 자리를 비워두는 함수
        katok[i-1] = katok[i]
        katok[i] = None

    del(katok[Klen-1])


## 전역 변수
katok = []
select = -1

## 메인 코드 부분
if __name__ == "__main__":

    while (select != 4):

        select = int(input('선택 --> 1:추가, 2: 삽입, 3:삭제 4:종료'))

        if(select == 1):
            data = input('추가할 데이터: ')
            add_data(data)
            print(katok)

        elif(select == 2):
            data = input('추가할 데이터: ')
            position = int(input('삽입할 위치:'))
            insert_data(position, data)
            print(katok)
        
        elif(select == 3):
            position = int(input('삭제할 위치'))
            delete_data(position)
            print(katok)
        
        elif(select == 4):
            print(katok)
            exit
        
        else:
            print('1 ~ 4까지의 정수만 입력하세요')
            continue
