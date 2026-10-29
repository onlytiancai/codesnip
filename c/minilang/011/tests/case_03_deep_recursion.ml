fn depth(n) {
    if (n <= 0) { return 0; }
    return 1 + depth(n - 1);
}
print(depth(100));
