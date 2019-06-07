
// utils

// 外部定义的话优先使用外部定义
var println = window.println || function (d) { console.log(d) };

function extractList(list, index) {
    return list.map(function (element) { return element[index]; });
}

function buildList(head, tail, index) {
    return [head].concat(extractList(tail, index));
}

function throwError(msg) {
    console.error(msg);
    throw msg;
}

// ast
var env = {};

// print
function PrintFunc() {
    this.type = 'func';
    this.params = ['o'];
    PrintFunc.prototype.eval = function (env) {
        println(env['o'].eval(env));
    }
}

env['print'] = PrintFunc;

function DefStat(name, params, body) {
    this.name = name;
    this.params = params;
    this.body = body;
    DefStat.prototype.eval = function (env) {
        funcs[this.name.id] = [this.params, this.body];
    }
}


function AssignStat(id, exp) {
    this.type = 'assignStat';
    this.id = id;
    this.exp = exp;
    AssignStat.prototype.eval = function (env) {
        env[this.id.id] = this.exp.eval(env);
    }
}


function CallExp(name, args) {
    this.type = 'CallExp';
    this.name = name;
    this.args = args;
    CallExp.prototype.eval = function (env) {
        console.log('funcs', funcs);

        var Func = env[this.name];

        if (!(Func && Func.type === 'func')) {
            throwError('Unknow call:' + this.name);
        }

        var func = new Func();

        // 形参和实参个数校验
        if (this.args.length != func.params.length) {
            throwError('args length error:' + func.params.length + ', ' + this.args.length);
        }

        // 构建实参
        var env = {};
        for (var i = 0; i < func.params.length; i++) {
            env[func.params[i]] = this.args[i];
        }

        func.eval(env);
    }
}

function SeqStat(body) {
    this.type = 'SeqStat';
    this.body = body;
    SeqStat.prototype.eval = function (env) {
        if (this.body.length) {
            for (var i = 0, len = this.body.length; i < len; i++) {
                this.body[i].eval(env);
            }
        }
    }
}

function WhileStat(cond, body) {
    this.type = 'WhileStat';
    this.cond = cond;
    this.body = body;
    WhileStat.prototype.eval = function (env) {
        while (this.cond.eval(env)) {
            this.body.eval(env);
        }
    }
}

function IfStat(cond, body) {
    this.type = 'IfStat';
    this.cond = cond;
    this.body = body;
    IfStat.prototype.eval = function (env) {
        if (this.cond.eval(env)) {
            this.body.eval(env);
        }
    }
}


function Integer(n) {
    this.type = 'Integer';
    this.n = n;
    Integer.prototype.eval = function (env) { return parseInt(this.n, 10); }
}

function Id(id) {
    this.type = 'Id';
    this.id = id;
    Id.prototype.eval = function (env) { return env[this.id]; }
}

function BinOpExp(left, op, right) {
    this.type = 'BinOpExp';
    this.op = op;
    this.left = left;
    this.right = right;
    BinOpExp.prototype.eval = function (env) {
        switch (this.op) {
            case '+': return this.left.eval(env)() + this.right.eval(env);
            case '-': return this.left.eval(env) - this.right.eval(env);
            case '*': return this.left.eval(env) * this.right.eval(env);
            case '/': return this.left.eval(env) / this.right.eval(env);
            case '%': return this.left.eval(env) % this.right.eval(env);
            case '<': return this.left.eval(env) < this.right.eval(env);
            case '>': return this.left.eval(env) > this.right.eval(env);
            case '<=': return this.left.eval(env) <= this.right.eval(env);
            case '>=': return this.left.eval(env) >= this.right.eval(env);
            case '==': return this.left.eval(env) == this.right.eval(env);
            case '!=': return this.left.eval(env) != this.right.eval(env);
            case 'and': return this.left.eval(env) && this.right.eval(env);
            case 'or': return this.left.eval(env) || this.right.eval(env);
            default:
                console.log('Unknow op:' + this.op)
                throw 'Unknow op:' + this.op;
        }
    }
} 