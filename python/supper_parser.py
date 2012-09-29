# -*- coding:utf-8 -*-
"""
一个通用词法分析器，不要求把所有的token都分析的特别精确
只要求把基本的词法单元分离出来，尽量兼容更多的语言
"""

import re

patt_tokens = r"""
(?P<c_comment_line>(?:\/\/).*)                         # c的单行注释
|(?P<js_regex>\/(?:\\\/)*.*?\/[a-zA-Z]?)                  # js的正则表达式字面量
|(?P<c_comment_mutil>\/\*[\s\S]*?\*\/)                  # c的多行注释
|(?P<python_string_mutil>:\'\'\'[\s\S]*?\'\'\')         #python多行字符串
|(?P<python_string_mutil2>\"\"\"[\s\S]*?\"\"\")         #python多行字符串
|(?P<python_comment>\#.*)                               # python shell的单行注释
|(?P<operator2>(?:\+\+)                                 # 多字符运算符
    |(?:\-\-)
    |(?:\-\>)
    |(?:\>\>)
    |(?:\<\<)
    |(?:\>\=)
    |(?:\<\=)
    |(?:\!\=)
    |(?:\+\=)
    |(?:\-\=)
    |(?:\=\=)
    |(?:\*\=)
    |(?:\/\=)
    |(?:\%\=)
    |(?:\&\=)
    |(?:\<\<\=)
    |(?:\>\>\=)
    |(?:\?\?)
    |(?:\?\:)
 ) 
|(?P<operator1>[\+\-!\.\(\)\[\]\{\}~&\/\*%\<\>\|\?=:])  # 单字符运算符
|(?P<number>\d+(?:\.\d+)?(?:[Ee]+[\+\-]?\d+)?[a-zA-Z]?) # 数字类型
|(?P<identity>[a-zA-Z_]+\w*)                            # 标识符
|(?P<string1>\"(?:(?:\\\")*?[^\"])*\")                   # 字符串
|(?P<string2>\'(?:(?:\\\')*?[^\'])*\')                   # 字符串
"""
re_tokens = re.compile(patt_tokens, re.VERBOSE)

token_type_map = dict(c_comment_mutil='comment',
                      python_string_mutil='comment',
                      python_string_mutil2='comment',
                      c_comment_line='comment',
                      python_comment='commen',
                      string1='string',
                      string2='string',
                      operator1='operator',
                      operator2='operator'
                      )


def _get_token_type(result):
    groupdict = result.groupdict().items()
    token_type = filter(lambda d: d[1] != None, groupdict)[0][0]
    return token_type_map.get(token_type, token_type)


def get_tokens(input):
    for result in re_tokens.finditer(input):
        yield _get_token_type(result), result.group()

if __name__ == '__main__':
    inputfile = None
    try:
        import sys
        inputfile = sys.argv[1]
    except:
        inputfile = __file__

    tokens = get_tokens(open(inputfile).read())
    for token_type, token in tokens:
        print "[[ %s %s ]]" % (token_type, token)
