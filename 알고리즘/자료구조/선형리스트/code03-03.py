# 데이터 삭제
katok = ['다현', '정연', '쯔위','미나', '사나', '지효', '모모']

def delete_data(position):
    Klen = len(katok) # katok배열의 길이를 저장할 변수 저장
    katok[position] = None

    for i in range(position+1, Klen, 1): # 삭제한 데이터 자리를 비워두는 함수
        katok[i-1] = katok[i]
        katok[i] = None

    del(katok[Klen-1])

delete_data(4)
print(katok)