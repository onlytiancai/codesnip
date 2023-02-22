function run(n) {
  for (var i = 2; i <= n; i++) {
    found = false;
    for (var j = 2; j < i; j++) {
      if (i % j == 0) {
        found = true;
        break;
      }
    }
    if (!found) console.log(i);
  }
}

run(100)
