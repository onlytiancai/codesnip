{

  // utils
    


  // 外部定义的话优先使用外部定义
  var println = window.println || function (d) { console.log(d) };

  function extractList(list, index) {
    return list.map(function(element) { return element[index]; });
  }

  function buildList(head, tail, index) {
    return [head].concat(extractList(tail, index));
  } 
  
  // ast
  var env = {};
  
  function AssignStat(id, exp) {
  	this.type = 'assignStat';
  	this.id = id;
    this.exp = exp;
    AssignStat.prototype.eval = function() {
    	env[this.id.id] = this.exp.eval(); 
    }    
  }
  
  function CallExp(name, args) {
  	this.type = 'CallExp';
  	this.name = name;
    this.args = args;
    CallExp.prototype.eval = function() {
    	switch (this.name) {
        	case 'print': println(this.args.eval());break;
        	default: throw 'Unknow call:' + this.name;
        }
    }
  }
  
  function SeqStat(body) {
  	this.type = 'SeqStat';
    this.body = body;
    SeqStat.prototype.eval = function() {
    	if (this.body.length) {
        	for(var i = 0, len = this.body.length; i < len; i ++) {
            	this.body[i].eval();
            }
        }
    }
  }
  
  function WhileStat(cond, body) {
  	this.type = 'WhileStat';
  	this.cond = cond;
    this.body = body;
    WhileStat.prototype.eval = function() {    	
    	while (this.cond.eval()) {
        	this.body.eval();
        }
    }
  }
  
  function IfStat(cond, body) {
  	this.type = 'IfStat';
  	this.cond = cond;
    this.body = body;
    IfStat.prototype.eval = function() {    	
    	if (this.cond.eval()) {
        	this.body.eval();	
        }
    }
  }
   
  function Integer(n) {
  	this.type = 'Integer';
    this.n = n;
  	Integer.prototype.eval = function() {return parseInt(this.n, 10);}
  }
  
  function Id(id) {
    this.type = 'Id';
    this.id = id;
  	Id.prototype.eval = function() {return env[this.id]; }
  }
  
  function BinOpExp(left, op, right) {  
  	this.type = 'BinOpExp';
    this.op = op;
    this.left = left;
    this.right = right;
    BinOpExp.prototype.eval = function() {
    	switch (this.op) {
        	case '+': return this.left.eval() + this.right.eval();
            case '-': return this.left.eval() - this.right.eval();
            case '*': return this.left.eval() * this.right.eval();
            case '/': return this.left.eval() / this.right.eval();
            case '%': return this.left.eval() % this.right.eval();
            case '<': return this.left.eval() < this.right.eval();
            case '>': return this.left.eval() > this.right.eval();
            case '<=': return this.left.eval() <= this.right.eval();
            case '>=': return this.left.eval() >= this.right.eval();            
            case '==': return this.left.eval() == this.right.eval();
            case '!=': return this.left.eval() != this.right.eval();
            case 'and': return this.left.eval() && this.right.eval();
            case 'or': return this.left.eval() || this.right.eval();            
        	default: 
            	console.log('Unknow op:' + this.op)
            	throw 'Unknow op:' + this.op;            
        }
    }      
  }   
}

Start  = __ body:SourceElements? __ { 	
	return { type: "Program", body: body}
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

    
AssignStat
	= id:Id _ '=' _ exp:Exp EOS { return new AssignStat(id, exp); }
    
PrintStat
	=  'print' _ exp:Exp { return new CallExp('print', exp) } 
    
IfStat
	= 'if'i _ cond:RelExp _ 'then'i EOS
    __ body:(SourceElements?) __
    'end'i  EOS { return new IfStat(cond, body)   }
    
WhileStat
	= 'while'i _ cond:RelExp _ 'then'i EOS
    __ body:(SourceElements?) __
    'end'i  EOS { return new WhileStat(cond, body)  }  
  
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
  = head:MathExp tail:(_ ("<=" / "<>"  / ">=" / "<" / ">" / "==" ) _ MathExp)*  _{
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

Factor
  = "(" _ expr:MathExp _ ")" { return expr; }
  / Integer
  / Id

              