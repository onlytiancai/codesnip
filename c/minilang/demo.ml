fn factorial(n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

let sum = 0;

for (let i = 1; i <= 5; i = i + 1) {
    sum = sum + i;
}

let result = factorial(5);
let root = sqrt(81);

print("sum = {sum}\n");
print("factorial = {result}\n");
print("sqrt = {root}\n");
