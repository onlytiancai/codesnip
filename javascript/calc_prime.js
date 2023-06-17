function check(i){
  for (var j = 2; j < i; j++) {
    if (i % j == 0) return true;
  }
  return false;

}
function run(n) {
  for (var i = 2; i <= n; i++) {
    if (!check(i)) console.log(i);
  }
}
run(30)
