# -*- coding: utf-8 -*-

import ply.lex as lex
import ply.yacc as yacc
import re
import sqlalchemy

tokens = (
    'TIMERANGE',  # 时间范围
    'CONDITION',  # 查询条件
    'DATA',       # 所操作的数据
    'QUERYDATA'   # 所取出的数据
)

t_TIMERANGE = r'(\d月份)|(今天)|(昨天)|(本周)|(上周)'
t_CONDITION = r'新增'
t_DATA = r'员工'
t_QUERYDATA = r'(总数)|(列表)'

t_ignore = ' \t\n'


def t_error(t):
    print "Illegal character '%s'" % t.value[0]
    t.lexer.skip(1)

lexer = lex.lex(reflags=re.UNICODE)


def p_expression_plus(p):
    'expression : TIMERANGE CONDITION DATA QUERYDATA'

    engine = sqlalchemy.create_engine('sqlite:///:memory:', echo=True)
    metadata = sqlalchemy.MetaData()
    emps = sqlalchemy.Table('emps', metadata,
            sqlalchemy.Column('name', sqlalchemy.String),
            sqlalchemy.Column('created_on', sqlalchemy.Date),
            )
    metadata.create_all(engine)

    str_timerange, str_condition, str_table, str_querydata = p[1], p[2], p[3], p[4]

    if str_table == '员工':
        if str_querydata == '总数':
            sql = sqlalchemy.select([sqlalchemy.func.count(emps.c.name)])
        elif str_querydata == '列表':
            sql = sqlalchemy.select([emps])

    if str_condition == '新增':
        sql = sql.where(emps.c.created_on == str_timerange)

    print sql

parser = yacc.yacc()

data = '''2月份新增员工总数'''
result = parser.parse(data)
