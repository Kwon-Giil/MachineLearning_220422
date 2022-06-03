## 함수
class TreeNode() :
    def __init__(self):
        self.left = None
        self.data = None
        self.right = None

## 전역
memory = []
root = None
nameAry = ['블랙핑크', '레드벨벳', '마마무', '에이핑크', '걸스데이', '트와이스']

## 메인
# 루트 노드 만들기
node = TreeNode()
node.data = nameAry[0]
root = node
memory.append(node)

for name in nameAry[1:] : # ['레드벨벳', '마마무',.... ]
    node = TreeNode()
    node.data = name

    current = root # root 10이고, name이 6
    while True :
        if name < current.data :
            if current.left == None :
                current.left = node
                break
            current = current.left
        else :
            if current.right == None :
                current.right = node
                break
            current = current.right

    memory.append(node)

print('이진 탐색 트리 구성 완료!')

findData = '마마무'

current = root
while True :
    if findData == current.data :
        print(findData, " 찾음 ^^")
        break
    elif findData < current.data :
        if current.left == None :
            print(findData, " 없음 ㅠㅠ")
            break
        current = current.left
    else :
        if current.right == None :
            print(findData, " 없음 ㅠㅠ")
            break
        current = current.right