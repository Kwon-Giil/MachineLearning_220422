## 함수
# 노드 정의

class Node() :
    def __init__(self):
        self.data = None
        self.link = None

# 노드 출력 함수
def printNodes(start) :
    current = start
    print(current.data, end='  ')
    while (current.link != None):
        current = current.link
        print(current.data, end='  ')
    print()

# 노드 삽입
def insertNode(findData, insertDate) :
    global  memory, head, pre, current
    # 첫 노드 앞에 삽입....  예 : 다현, 화사
    if findData == head.data :
        node = Node()
        node.data = insertDate
        node.link = head
        head = node
        memory.append(node)
        return
    
    # 중간 노드 삽입.... 예 : 사나, 솔라
    current = head
    while current.link != None :
        pre = current
        current = current.link
        if current.data == findData :
            node = Node()
            node.data = insertDate
            node.link = current
            pre.link = node
            memory.append(node)
            return
    
    # 마지막 노드 삽입 (=findData 없음).. 예: 재남, 문별
    node = Node()
    node.data = insertDate
    current.link = node
    memory.append(node)
    return

def deleteNode(deleteData):
    global  memory, head, pre, current

    # 첫번째 노드 삭제
    if deleteData == head.data:
        current = head
        head = head.link
        del(current)
        return

    # 2번째 이후 노드 삭제
    current = head
    while current != None:
        pre = current
        current = current.link

        if current.data == deleteData:
            pre.link = current.link
            del(current)
            return

def findNode(findData):
    global  memory, head, pre, current
    current = head
    
    # 첫번째 노드를 검색
    if current.data == findData:
        return current

    # 두번째 이후 노드 검색
    while current != None:
        current = current.link

        if current.data == findData:
            return current
    return Node()


## 전역
memory = []
head, pre, current = None, None, None
dataArray = ['다현', '정연', '쯔위', '사나', '지효']

## 메인
node = Node() # 첫 노드
node.data = dataArray[0]
head = node
memory.append(node)

for data in dataArray[1:] :  # ['정연', '쯔위', ..... ]
    pre = node
    node = Node()
    node.data = data
    pre.link = node
    memory.append(node)

printNodes(head)

#insertNode('다현', '나연')

#insertNode('사나', '모모')

#insertNode('채영', '미나')
#printNodes(head)

#deleteNode('다현')
#printNodes(head)

#deleteNode('쯔위')
#printNodes(head)

fNode = findNode('쯔위')
print(fNode.data)