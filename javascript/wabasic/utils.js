var utils = (function() {
    function Set(list) {
        console.log('Set#__construct', list);
        this.list = list || [];
        Set.prototype.contains = function(o) {
            return this.list.indexOf(o) != -1;
        }
        Set.prototype.add = function(o) {
            console.log('Set#add', o);
            if (!this.contains(o)) {
                this.list.push(o);
            }
            return this;
        }
        Set.prototype.intersect = function(set) {
            var ret = [];
            for (var i = 0; i < this.list.length; i++) {
                var x = this.list[i];
                for (var j = 0; j < set.list.length; j++) {
                    var y = set.list[j];
                    if (x === y) ret.push(x);
                }
            }
            return ret;
        }
    }

    return {
        Set: Set,
    };
}());