from logsql import parse, query
import unittest

class StatTest(unittest.TestCase):
    log = '''10:11 111 222
10:12 333 444
10:14 333 666
10:13 555 666'''
    rule = 'a b c'
    def test_filter(self):
        actual = query(self.log.splitlines(), self.rule, ['*'], {'b': '333'}) 
        expected = ['10:12 333 444', '10:14 333 666']
        self.assertListEqual(actual, expected)

        actual = query(self.log.splitlines(), self.rule, ['*'], {'b': '333', 'c': '444'}) 
        expected = ['10:12 333 444']
        self.assertListEqual(actual, expected)

    def test_group_count(self):
        actual = query(log=self.log.splitlines(), 
                       rule=self.rule, 
                       select=['b','count(*)'],
                       group='b'
                       )
        expected = ['111 1', '333 2', '555 1']
        self.assertListEqual(actual, expected)

class ParseTest(unittest.TestCase):
    '''根据规则解析文本日志为 json，以便对日志进行按字段的过滤统计
    - 解析规则的多字段用空格隔开
    - 字段支持包裹字符以支持字段内有空格或引号的情况'''

    def test_error_token(self):
        with self.assertRaisesRegex(Exception, 'error token:'):
            parse('a "b c', '')

    def test_base(self):
        rule = 'a  b c'
        line = '111 222 333'
        expected = {'a':'111', 'b':'222', 'c': '333'} 
        self.assertDictEqual(parse(rule, line), expected)

    def test_enclose(self):
        rule = '[a]  "b" c'
        line = '[1"11] "2 22" 333'
        expected = {'a':'1"11', 'b':'2 22', 'c': '333'} 
        self.assertDictEqual(parse(rule, line), expected)

    def test_escape(self):
        rule = 'a "b" [c]'
        line = r'111 "2\" 66 \"" [3\]33]'
        expected = {'a': '111', 'b':r'2\" 66 \"', 'c': r'3\]33'} 
        self.assertDictEqual(parse(rule, line), expected)

    def test_skip_field(self):
        rule = 'a - b'
        line = r'111 222 333'
        expected = {'a': '111', 'b':'333'} 
        self.assertDictEqual(parse(rule, line), expected)


if __name__ == '__main__':
    unittest.main()
