function calc_prime(n) {
  for (var i = 2; i <= n; i++) {
    var found = false;
    for (var j = 2; j < i; j++) {
      found = i % j == 0 ? true : found;
    }
    found || console.log(i);
  }
}

calc_prime(30)
