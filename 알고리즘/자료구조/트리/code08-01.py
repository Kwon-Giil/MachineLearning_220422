## 함수 부분
class Tree_Node():
    def __init__(self) -> None:
        self.left = None
        self.data = None
        self.right = None

## 변수 부분


## 기능
node1 = Tree_Node()
node1.data = '화사'

node2 = Tree_Node()
node2.data = '솔라'
node1.left = node2

node3 = Tree_Node()
node3.data = '문별'
node1.right = node3

node4 = Tree_Node()
node4.data = '휘인'
node2.left = node4

node5 = Tree_Node()
node5.data = '쯔위'
node2.right = node5

node6 = Tree_Node()
node6.data = '선미'
node3.left = node4

print(node1.data)
print(node1.left.data, '\t', node1.right.data)
print(node1.left.left.data, '\t', node1.left.right.data, '/t', node1.right.left.data)