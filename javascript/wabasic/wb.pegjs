{
    // ****** 辅助函数

    // 外部定义的话优先使用外部定义
    var println = window.println || function (d) { console.log(d) };

    /**
     * 工具方法：构建顺序语句列表使用
     * extractList([[1,2], [2, 3], [4,5]], 1) => [2, 3, 5]
     */
    function extractList(list, index) {
        return list.map(function (element) { return element[index]; });
    }

    /**
     * 工具方法：构建顺序语句列表使用
     * buildList(100, [[1,2], [3,4],[5,6]], 1) => [100, 2, 4, 6]
     */
    function buildList(head, tail, index) {
        return [head].concat(extractList(tail, index));
    }

    /**
     * 抛出语法或语义错误 
     */
    function throwError(msg) {
        console.error(msg);
        throw msg;
    }

    // ****** 构建 ast
    var env = {};

    /**
     * 内置 print 函数
     */
    var PrintFunc = {
        type: 'func',
        params: 'o',
        eval: function (env) {
            console.log('PrintFunc#eval', env);
            println(env['o']);
        }
    };

    /**
     * 函数定义语句
     * @param {函数名} name 
     * @param {形参列表} params 
     * @param {函数体} body 
     */
    function DefStat(name, params, body) {
        this.type = 'def';
        this.name = name;
        this.params = params;
        this.body = body;
        DefStat.prototype.eval = function (env) {
            console.log('DefStat#eval', env, this.params);
            var that = this;
            env[this.name.id] = {
                // 实现闭包：捕获变量
                capture: Object.assign({}, env),
                type: 'func',
                params: this.params,
                eval: function (env) {
                    console.log('Func#eval' + that.name.id, env);
                    that.body.eval(env);
                    // 实现闭包：返回修改后的环境
                    return env;
                }
            };
        }
    }

    /**
     * 赋值语句
     * @param {变量名} id 
     * @param {表达式} exp 
     */
    function AssignStat(id, exp) {
        this.type = 'assignStat';
        this.id = id;
        this.exp = exp;
        AssignStat.prototype.eval = function (env) {
            console.log('AssignStat#eval' + this.id.id, env);
            env[this.id.id] = this.exp.eval(env);
        }
    }

    /**
     * return 语句使用 throw 实现，这里封装要 throw 的对象
     * @param {返回的对象} ret 
     */
    function ReturnObject(ret) {
        this.type = 'ReturnObject'
        this.ret = ret;
    }

    /**
     * return 语句
     * @param {返回的表达式} exp 
     */
    function ReturnStat(exp) {
        this.type = 'ReturnStat';
        this.exp = exp;
        ReturnStat.prototype.eval = function (env) {
            console.log('ReturnStat:eval', env, exp);
            throw new ReturnObject(this.exp.eval(env));
        }
    }

    /**
     * 函数调用表达式
     * @param {要调用的函数名} name 
     * @param {实参列表} args 
     */
    function CallExp(name, args) {
        this.type = 'call';
        this.name = name;
        this.args = args;
        CallExp.prototype.eval = function (env) {
            console.log('CallExp:eval before:', this.name.id, env);

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

            console.debug('CallExp:capture, env', this.name.id, func.capture, env);
            // 实现闭包：合并定义期间捕获的变量
            env = Object.assign(func.capture || {}, env);

            // 构建实参
            for (var i = 0; i < func.params.length; i++) {
                env[func.params[i]] = this.args[i].eval(env);
            }

            try {
                // 再次拷贝，防止递归调用时外层函数的变量被内层函数改动
                var modifyenv = func.eval(Object.assign({}, env));
                // 实现闭包：修改捕获的变量            
                var keys = Object.keys(func.capture || {});
                for (var i = 0; i < keys.length; i++) {
                    var key = keys[i];
                    func.capture[key] = modifyenv[key];
                }
            } catch (err) {
                // 使用 thorw 实现 return 语句
                if (err instanceof ReturnObject) {
                    return err.ret;
                }
                throw err;

            }
            console.log('CallExp:eval after:', this.name.id, env);
        }
    }


    /**
     * 顺序语句
     * @param {语句列表} body 
     */
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

    /**
     * 循环语句：while
     * @param {条件} cond 
     * @param {主体} body 
     */
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

    /**
     * 分支语句：if
     * @param {条件} cond 
     * @param {主体} body 
     */
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

    /**
     * 整形数字
     * @param {数字} n 
     */
    function Integer(n) {
        this.type = 'Integer';
        this.n = n;
        Integer.prototype.eval = function (env) { return parseInt(this.n, 10); }
    }

    /**
     * 标识符
     * @param {名称} id 
     */
    function Id(id) {
        this.type = 'Id';
        this.id = id;
        Id.prototype.eval = function (env) { return env[this.id]; }
    }

    /**
     * 程序入口
     * @param {主体} body 
     */
    function Program(body) {
        this.type = 'Program';
        this.body = body;
        Program.prototype.eval = function () {
            var env = {};
            env['print'] = PrintFunc;
            this.body.eval(env);
        }
    }

    /**
     * 二元操作符
     * @param {left} left 
     * @param {op} op 
     * @param {right} right 
     */
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
Id = !Keyword ([a-z]+ [0-9]* [z-z]*)  { return new Id(text())}
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
	=  'print' ' '+ exp:Exp { return new CallExp(new Id('print'), [exp]) } 

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

              