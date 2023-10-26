from logsql import parse
import unittest

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

    def test_nginx_log(self):
        rule = 'time status_code:int request_time:float upstream_time:float - ip "x_forward" host socket "url" bytes "-" "agent"'
        line = '2023-10-26T16:19:44+08:00 200 0.021 0.020 - 172.31.24.99 "49.37.43.52, 172.70.184.133" abc.com unix:/var/run/php/php8.1-fpm.sock "GET /api/users HTTP/1.1" 731 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"'
        expected = {'time': '2023-10-26T16:19:44+08:00', 
                    'status_code': 200, 
                    'request_time': 0.021,
                    'upstream_time': 0.020,
                    'ip': '172.31.24.99',
                    'x_forward': '49.37.43.52, 172.70.184.133',
                    'host': 'abc.com',
                    'socket': 'unix:/var/run/php/php8.1-fpm.sock',
                    'url': 'GET /api/users HTTP/1.1',
                    'bytes': '731',
                    'agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36' 
                   } 
        self.assertDictEqual(parse(rule, line), expected)


if __name__ == '__main__':
    unittest.main()
