### 노드 함수 정의
class Node() :
    def __init__(self):
        self.data = None
        self.link = None

### 메인(노드 생성)
node1 = Node()
node1.data = '다현'

node2 = Node()
node2.data = '정연'
node1.link = node2

node3 = Node()
node3.data = "쯔위"
node2.link = node3

node4 = Node()
node4.data = "사나"
node3.link = node4

node5 = Node()
node5.data = "지효"
node4.link = node5

# 노드 삭제
node2.link = node3.link # 삭제하기 전 node2의 링크를 변경
del(node3) # 노드3 삭제

# 새로운 노드 삽입할 때
# newNode = Node()
# newNode.data = '미나'
# newNode.link = node2.link
# node2.link = newNode

current = node1
print(current.data, end='  ') # 시작 노드 출력
while (current.link != None) :
    current = current.link
    print(current.data, end='  ') # 시작 노드에 링크된 데이터 출력