list = (v, l) => (type) => type === 'v' ? v : l;
head = l => l('v');
tail = l => l('t');
const l = list(1, list(2, list(3, null)));

console.log(head(l));
console.log(head(tail(l)));
console.log(head(tail(tail(l))));
console.log(tail(tail(tail(l))));
