def get_prime(n):
    for i in range(int(pow(n,0.5))+1): 
        yield i

print(list(get_prime(100)))
