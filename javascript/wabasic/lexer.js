var lexer = (function() {
    var Set = utils.Set;

    /**
     * NFA 规则
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
     * 
     * @param {可接受状态集} acceptStates 
     * @param {规则集} rules 
     */
    function Nfa(name, acceptStates, rules) {
        this.name = name;
        this.state = new Set([0]);
        this.acceptStates = new Set(acceptStates);
        this.rules = rules;
        this.lastMatchedIndex = -1;

        Nfa.prototype.parse = function(input) {
            var chars = input.split('');

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

                console.log('lexer.Nfa#parse', this.name, i, char, dstStates.list);

                this.state = dstStates;
                // 更新最后匹配成功的位置
                if (this.accepted()) {
                    this.lastMatchedIndex = i;
                }

            }

            return this;
        }

        Nfa.prototype.accepted = function() {
            console.log('lexer.Nfa#accepted', this.name, this.acceptStates.list, this.state.list);
            return this.acceptStates.intersect(this.state).length > 0;
        }
    }

    /**
     * \d+
     */
    function newNum() {
        return new Nfa('Num', [1], [
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
        ]);
    }

    /**
     * [\+\-\*\/]
     */
    function newArithmeticOp() {
        return new Nfa('ArithmeticOp', [1], [
            new Rule(0, '+', 1),
            new Rule(0, '-', 1),
            new Rule(0, '*', 1),
            new Rule(0, '/', 1),
        ]);
    }

    /**
     * \s+
     */
    function newWhiteSpace() {
        return new Nfa('WhiteSpace', [1], [
            new Rule(0, ' ', 1),
            new Rule(0, '\t', 1),
            new Rule(1, ' ', 1),
            new Rule(1, '\t', 1),
        ]);
    }

    /**
     * "\d+"
     */
    function newString() {
        return new Nfa('String', [2], [
            new Rule(0, '"', 1),
            new Rule(1, '0', 1),
            new Rule(1, '"', 2),
            new Rule(0, '"', 2),
        ]);
    }

    function next(input) {
        var nfas = [newNum, newArithmeticOp, newWhiteSpace];
        var maxMatchedIndex = -1;
        for (var i = 0; i < nfas.length; i++) {
            var nfa = nfas[i]();
            nfa.parse(input);
            console.log('lexer.next:(' + nfa.name + ') lastMatchedIndex=' + nfa.lastMatchedIndex);
            if (nfa.lastMatchedIndex > maxMatchedIndex) {
                maxMatchedIndex = nfa.lastMatchedIndex;
            }
        }
        return maxMatchedIndex > -1 ? input.substr(0, maxMatchedIndex + 1) : null;
    }

    var parse = function(input) {
        var ret = [];
        var remain = input;
        do {
            var token = next(remain);
            if (token != null) {
                ret.push(token);
                remain = remain.substr(token.length);
                console.log('lexer.parse:' + '[' + token + '](' + remain + ')');
            } else {
                throw "Unknow input:" + remain;
            }
        }
        while (remain.length > 0)
        return ret;
    }

    return {
        parse: parse,
        newNum: newNum,
        newArithmeticOp: newArithmeticOp,
        newString: newString,
    }
}());