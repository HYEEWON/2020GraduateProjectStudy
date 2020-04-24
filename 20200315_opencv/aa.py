t = int(input())
for i in range(t):
    p = input()
    p.replace("RR", '')
    n = int(input())
    arr = list(map(str, input()[1:-1].split(',')))

    D_front, D_back, d = 0, 0, True
    for j in p:
        if j == 'R':
            d = not d
        elif j == 'D':
            if d:
                D_front += 1
            else:
                D_back += 1

    if D_front + D_back <= n:
        arr = arr[D_front: n - D_back]
        if d:
            a = 1
        else:
            arr.reverse()
        if len(arr) == 0: print("[]")
        else:
            print('[' + arr[0], end='')
            for i in range(1, len(arr)):
                print(',' + arr[i], end='')
            print(']')
    else:
        print("error")
