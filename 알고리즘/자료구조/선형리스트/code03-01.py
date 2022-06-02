# 카톡 온 순서대로 친구 순서 정렬하기
katok = []

## 친구 추가 함수
def  add_data(friend) :
    katok.append(None)
    kLen = len(katok)
    katok[kLen-1] = friend

# 친구 추가
add_data('다현')
add_data('정연')
add_data('쯔위')
add_data('사나')
add_data('지효')
add_data('모모')
print(katok)

