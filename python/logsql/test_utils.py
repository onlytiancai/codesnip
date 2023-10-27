from logsql import format_time 
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


     
