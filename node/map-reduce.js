// To run this example install levelup and leveldown. 
// npm install levelup leveldown 
// please delete /tmp/map-reduce-example on each run 

var db = require('level-sublevel')
  (require('levelup')('/tmp/map-reduce-example'))
var MapReduce = require('map-reduce')

var mapped = MapReduce(db, 'example', function (key, value, emit) {
  console.log('MAP', key, value, '->', Number(value) % 2 ? 'odd' : 'even')
  if(Number(value) % 2 == 0)
    emit('even', Number(value))
  else
    emit('odd', Number(value))
}, function (acc, v) {
  return Number(acc || 0) + Number(v)
})

db.put('a', '1', console.log)
db.put('b', '2', console.log)
db.put('c', '3', console.log)
db.put('d', '4', console.log)
db.put('e', '5', function () {

  mapped.on('reduce', function (group, val) {
    console.log('Reduce', group, val)
  })

})

