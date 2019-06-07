{

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
var PrintFunc = {
    type: 'func',
    params: 'o',
    eval: function(env) {
        console.log('PrintFunc#eval', env);      
        println(env['o']);
    }
};

function DefStat(name, params, body) {
    this.type = 'def';
    this.name = name;
    this.params = params;
    this.body = body;
    DefStat.prototype.eval = function (env) {
        console.log('DefStat#eval', env, this.params);
        var that = this;
        env[this.name.id] = {
            type: 'func',
            params: this.params,
            eval: function(env) {
                console.log('Def#' + that.name.id, env);
                that.body.eval(env);
            }
        };        
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
    this.type = 'call';
    this.name = name;
    this.args = args;
    CallExp.prototype.eval = function (env) {
        console.log('CallExp:before:', this.name.id, env);
        
        var func = env[this.name.id];
        
        if (!func) {
            throwError('Unknow func:' + this.name);
        }

        if (func.type != 'func') {
            throwError('Id is not func:' + this.name);
        }

        // 形参和实参个数校验
        if (this.args.length != func.params.length) {
            throwError('args length error:' + func.params.length + ', ' + this.args.length);
        }

        // copy env
        env = Object.assign({}, env);
        // 构建实参
        for (var i = 0; i < func.params.length; i++) {           
            env[func.params[i]] = this.args[i].eval(env);
        }        

        func.eval(env);
        console.log('CallExp:after:', this.name.id, env);
        return env['__return'];
    }
}

function ReturnStat( exp) {
    this.type = 'ReturnStat';
    this.exp = exp;
    ReturnStat.prototype.eval = function (env) {    
        console.log('ReturnStat:', env, exp);    
        env['__return'] = this.exp.eval(env);
    }
}

function SeqStat(body) {
    this.type = 'SeqStat';
    this.body = body;
    SeqStat.prototype.eval = function (env) {
        console.log('SeqStat:env', env);
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

function Program(body) {
    this.type = 'Program';
    this.body = body;
    Program.prototype.eval = function() {
        var env = {};
        env['print'] = PrintFunc;
        this.body.eval(env);
    }
}

function BinOpExp(left, op, right) {
    this.type = 'BinOpExp';
    this.op = op;
    this.left = left;
    this.right = right;
    BinOpExp.prototype.eval = function (env) {
        console.log('BinOpExp#eval', this.left, this.op, this.right);
        console.log('BinOpExp#eval', this.left.eval(env), this.op, this.right.eval(env));
        switch (this.op) {
            case '+': return this.left.eval(env) + this.right.eval(env);
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
}

Start  = __ body:SourceElements? __ { 	
	return new Program(body);
}

WhiteSpace "whitespace" = [\t ]
LineTerminator = [\n\r]
LineTerminatorSequence "end of line"  = "\n"  / "\r\n"  / "\r"
__  = (WhiteSpace / LineTerminatorSequence / Comment)*
_  = (WhiteSpace)*
EOF = !. 
EOS  = __ ";" / _ Comment? LineTerminatorSequence   / __ EOF  
Comment  = "//" (!LineTerminator .)*  
Integer "integer"  = _ [0-9]+ { return new Integer(text()); }
Id = !Keyword ([a-z]+)  { return new Id(text())}
Keyword  = 'if' / 'then'  / 'end'  / 'while' / 'and' / 'or'
 
SourceElements
  = head:Statement tail:(__ Statement)* {
  		var body = buildList(head, tail, 1);
  		return new SeqStat(body);      
    }
    
Statement
  = AssignStat
  / PrintStat
  / IfStat
  / WhileStat  
  / DefStat
  / CallExp
  / ReturnStat

    
AssignStat
	= id:Id _ '=' _ exp:Exp EOS { return new AssignStat(id, exp); }
    
PrintStat
	=  'print' _ exp:Exp { return new CallExp(new Id('print'), [exp]) } 

ReturnStat
	=  'return' _ exp:Exp { return new ReturnStat(exp) }     
    
IfStat
	= 'if'i _ cond:RelExp _ 'then'i EOS
    __ body:(SourceElements?) __
    'end'i  EOS { return new IfStat(cond, body)   }
    
WhileStat
	= 'while' _ cond:RelExp _ 'then'i EOS
    __ body:(SourceElements?) __
    'end'i  EOS { return new WhileStat(cond, body)  }  
  
DefStat
	= 'def'i _ name:Id '(' _ params:ParamList _ ')' EOS
    __ body:(SourceElements?) __
    'end'i  EOS { return new DefStat(name, params, body)  }  

ParamList
    = head:Id _ ',' _ tail:ParamList { tail.unshift(head.id);return tail;}
    / id:Id { return [id.id]}  
    / _ {return []}

CallExp
	= _ name:Id '(' _ args:ArgList _ ')' { return new CallExp(name, args)  }      
      

ArgList
    = head:Exp _ ',' _ tail:ArgList { tail.unshift(head);return tail;}
    / exp:Exp { return [exp]}  
    / _ {return []} 

Exp
	= exp:OrExp { return exp }
    
OrExp
  = head:AndExp tail:(_ ( "or") _ AndExp)* _ {
      return tail.reduce(function(result, element) {
		return new BinOpExp(result, element[1], element[3])
      }, head);
    }
    
AndExp
  = head:RelExp tail:(_ ("and") _ RelExp)*  _ {
      return tail.reduce(function(result, element) {
      	return new BinOpExp(result, element[1], element[3])
      }, head);
    }    
  
RelExp
  = head:MathExp tail:(_ ("<=" / "<>"  / ">=" / "<" / ">" / "==" / "!=" ) _ MathExp)*  _{
      return tail.reduce(function(result, element) {
      	return new BinOpExp(result, element[1], element[3])
      }, head);
    }    
    
MathExp
  = head:Term tail:(_ ( "+" / "-"  ) _ Term)* _ {
      return tail.reduce(function(result, element) {      	
        return new BinOpExp(result, element[1], element[3])
      }, head);
    }

Term
  = head:Factor tail:(_ ("*" / "/" / "%") _ Factor)* {
      return tail.reduce(function(result, element) {
		return new BinOpExp(result, element[1], element[3])
      }, head);
    }

// 注意：CallExp 要在 Id 上面
Factor
  = "(" _ expr:MathExp _ ")" { return expr; }
  / CallExp
  / Integer
  / Id  

              