v_in = input('input number : ')
n = int(v_in)
for i in range(0, n):
    if i <= n/2:
        print(' ' * int(n/2-i), '*' * int(2*i+1))
    else:
        print(' ' * int(i-n/2+1), '*' * int(2*(n-i)-1))