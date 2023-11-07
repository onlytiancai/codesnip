from logsql import format_time, _split_select
from datetime import datetime
import unittest

class UtilsTest(unittest.TestCase):
    def test_format_time(self):
        time = datetime.strptime('2023-10-01 08:06:21.123456', '%Y-%m-%d %H:%M:%S.%f')
        actual = format_time(time, '1h').strftime('%Y-%m-%d %H:%M:%S')
        expected = '2023-10-01 08:00:00'
        self.assertEqual(actual, expected)

        actual = format_time(time, '5m').strftime('%Y-%m-%d %H:%M:%S')
        expected = '2023-10-01 08:05:00'
        self.assertEqual(actual, expected)

        actual = format_time(time, '10m').strftime('%Y-%m-%d %H:%M:%S')
        expected = '2023-10-01 08:00:00'
        self.assertEqual(actual, expected)

        actual = format_time(time, '5s').strftime('%Y-%m-%d %H:%M:%S')
        expected = '2023-10-01 08:06:20'
        self.assertEqual(actual, expected)


    def test_split_select(self): 
        txt = 'foo(a,b(2,3),e),b,a(c(6),2)'
        actual = list(_split_select(txt))
        expected = ['foo(a,b(2,3),e)', 'b', 'a(c(6),2)']
        self.assertListEqual(actual, expected)
