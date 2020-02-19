const dataset = require('ml-dataset-iris').getNumbers();
const kmeans = require('ml-kmeans').default;

let data = dataset;

let ans = kmeans(data, 4, { initialization: 'random' });
console.log(111, ans);