## 함수
def isQFull():
    global SIZE, queue, front, rear
    if (rear != SIZE-1) :
        return False
    elif (rear == SIZE-1 and front == -1) :
        return True
    else :
        for i in range(front+1, SIZE, 1) :
            queue[i-1] = queue[i]
            queue[i] = None
        front -= 1
        rear -= 1
        return False

def enQueue(data):
    global SIZE, queue, front, rear
    if (isQFull()):
        print('큐가 꽉 찼습니다.')
        return
    rear += 1
    queue[rear] = data

def isQEmpty() :
    global SIZE, queue, front, rear
    if (front == rear) :
        return True
    else :
        return False

def deQueue() :
    global SIZE, queue, front, rear
    if (isQEmpty()) :
        print('큐 텅~~')
        return None
    front += 1
    data = queue[front]
    queue[front] = None
    return data

def peek() :
    global SIZE, queue, front, rear
    if (isQEmpty()) :
        print('큐 텅~~')
        return None
    return queue[front+1]


# 변수
SIZE = 5
queue = [None for _ in range(SIZE)]
front = rear = -1

# 메인 
enQueue('화사')
enQueue('솔라')
enQueue('문별')
enQueue('휘인')
enQueue('선미') # 원더걸스
print('출구<--', queue, '<--입구')

retData = deQueue()
print("디큐 ==> ", retData)
retData = deQueue()
print("디큐 ==> ", retData)
retData = deQueue()
print("디큐 ==> ", retData)
print('출구<--', queue, '<--입구')

enQueue('태연')
print('출구<--', queue, '<--입구')

enQueue('윤아')
print('출구<--', queue, '<--입구')

enQueue('지은')
print('출구<--', queue, '<--입구')

print("다음 예정 손님 =>", peek())
print('출구<--', queue, '<--입구')
