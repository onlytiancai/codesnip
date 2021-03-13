// construct a new list that holds the given value and points
// to the given list (the tail) O(1) constant
function list(x, l) {
  return function(h){
    return h ? x : l;
  }
} 

// return the value stored at the head of a list O(1) constant
function head(l) {
  return l(true);
}

// return the tail of a list O(1) constant
function tail(l) {
  return l(false);
}

// given a list, return the nth value down; calling this 
// with n = 0 is equivalent to using head O(n) linear
function nth(l, n) {
  return n == 0 ? head(l) : nth(tail(l), n - 1);
}

// given a list and a transform function, produce a new list with
// that transform applied to every value in the list O(n) linear
function map(l, f) {
  return l == null ? null : list(f(head(l)), map(tail(l), f))
}

var l = list(1, list(2, list(3, null)))
console.log(nth(l, 0), nth(l, 1), nth(l, 2))
l = map(l, function(x) {return x*2;})
console.log(nth(l, 0), nth(l, 1), nth(l, 2))
