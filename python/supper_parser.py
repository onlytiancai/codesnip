# -*- coding:utf-8 -*-
"""
一个通用词法分析器，不要求把所有的token都分析的特别精确
只要求把基本的词法单元分离出来，尽量兼容更多的语言
"""

import re

patt_tokens = r"""
(?P<comment1>(?:\/\/).*)                                # c的单行注释
|(?P<regex>\/(?:\\\/)*.*?\/[a-zA-Z]?)                   # js的正则表达式字面量
|(?P<comment2>\/\*[\s\S]*?\*\/)                         # c的多行注释
|(?P<string1>:\'\'\'[\s\S]*?\'\'\')                     # python多行字符串
|(?P<string2>\"\"\"[\s\S]*?\"\"\")                      # python多行字符串
|(?P<comment3>\#.*)                                     # python shell的单行注释
|(?P<operator1>(?:\+\+)                                 # 多字符运算符
    |(?:\-\-) |(?:\-\>) |(?:\>\>) |(?:\<\<) |(?:\>\=)
    |(?:\<\=) |(?:\!\=) |(?:\+\=) |(?:\-\=) |(?:\=\=)
    |(?:\*\=) |(?:\/\=) |(?:\%\=) |(?:\&\=) |(?:\<\<\=)
    |(?:\>\>\=) |(?:\?\?) |(?:\?\:)
 )
|(?P<operator2>[\+\-!\.\(\)\[\]\{\}~&\/\*%\<\>\|\?=:])  # 单字符运算符
|(?P<number>\d+(?:\.\d+)?(?:[Ee]+[\+\-]?\d+)?[a-zA-Z]?) # 数字类型
|(?P<identity>[a-zA-Z_]+\w*)                            # 标识符
|(?P<string3>\"(?:(?:\\\")*[^\"])*?\")                  # 字符串
|(?P<string4>\'(?:(?:\\\')*[^\'])*?\')                  # 字符串
"""
re_tokens = re.compile(patt_tokens, re.VERBOSE)

def _get_token_type(result):
    groupdict = result.groupdict().items()
    token_type = filter(lambda d: d[1] is not None, groupdict)[0][0]
    return token_type.rstrip('123456789')


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
