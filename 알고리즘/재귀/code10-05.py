def CntDwn(num):
    if num == 0: # 숫자를 다 세서 0이 되면 stop
        print('발사~~~!')
    else:
        print(num) # 0이 아닌 숫자 출력
        CntDwn(num-1) # 다시 돌아가서 숫자를 까는 거

print(CntDwn(10))