var lexer = (function() {
    var Set = utils.Set;

    /**
     * 
     * @param {起始状态} src 
     * @param {输入字符} char 
     * @param {目标状态} dst 
     */
    function Rule(src, char, dst) {
        this.src = src;
        this.char = char;
        this.dst = dst;
    }

    /**
     * \d+
     */
    function Num(input) {
        this.input = input;
        this.state = new Set([0]);
        this.acceptStates = new Set([1]);

        this.rules = [
            new Rule(0, '0', 1),
            new Rule(0, '1', 1),
            new Rule(0, '2', 1),
            new Rule(0, '3', 1),
            new Rule(0, '4', 1),
            new Rule(0, '5', 1),
            new Rule(0, '6', 1),
            new Rule(0, '7', 1),
            new Rule(0, '8', 1),
            new Rule(0, '9', 1),
            new Rule(1, '0', 1),
            new Rule(1, '1', 1),
            new Rule(1, '2', 1),
            new Rule(1, '3', 1),
            new Rule(1, '4', 1),
            new Rule(1, '5', 1),
            new Rule(1, '6', 1),
            new Rule(1, '7', 1),
            new Rule(1, '8', 1),
            new Rule(1, '9', 1),
        ];

        Num.prototype.parse = function() {
            var chars = this.input.split('');

            for (var i = 0; i < chars.length; i++) {

                var char = chars[i];
                var dstStates = new Set();
                for (var j = 0; j < this.rules.length; j++) {
                    var rule = this.rules[j];
                    // 当前状态包含规则起始状态，且当前字符是规则输入字符
                    if (this.state.contains(rule.src) && char == rule.char) {
                        dstStates.add(rule.dst);
                    }
                }
                console.log('Num#parse', char, dstStates.list);
                this.state = dstStates;

            }

            return this;
        }

        Num.prototype.accepted = function() {
            console.log('Num#accepted', this.acceptStates.list, this.state.list);
            return this.acceptStates.intersect(this.state).length > 0;
        }
    }

    function Plus() {

    }

    var parse = function(input) {
        /**
         * \d+
         * \+
         */
        var arr = input.split('');
        for (var i = 0; i < arr.length; i++) {
            const c = arr[i];
        }
    }

    return {
        parse: parse,
        Num: Num,
    }
}());