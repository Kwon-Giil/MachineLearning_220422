# 데이터 삽입
katok = ['다현', '정연', '쯔위', '사나', '지효', '모모']

def  insert_data(position, friend) :
    katok.append(None)
    kLen = len(katok)

# 데이터를 뒤에 밀어내고 중간에 빈칸 생성
    for i in range(kLen-1, position, -1) :
        katok[i] = katok[i-1]
        katok[i-1] = None
        
    katok[position] = friend


insert_data(3, '미나')
print(katok)