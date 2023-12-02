from logsql import select 
import unittest
from datetime import datetime

class SelectTest(unittest.TestCase):
    data = [{'time': datetime.strptime('2023-10-01 08:03', '%Y-%m-%d %H:%M'), 'gender': 'boy', 'age': 18},
            {'time': datetime.strptime('2023-10-01 08:15', '%Y-%m-%d %H:%M'), 'gender': 'boy', 'age': 20},
            {'time': datetime.strptime('2023-10-01 08:21', '%Y-%m-%d %H:%M'), 'gender': 'girl', 'age': 16},
            {'time': datetime.strptime('2023-10-01 09:15', '%Y-%m-%d %H:%M'), 'gender': 'girl', 'age': 18},
            {'time': datetime.strptime('2023-10-01 11:21', '%Y-%m-%d %H:%M'), 'gender': 'boy', 'age': 56},
           ] 

    def test_base_groupby(self):
        query = select('gender,avg(age),min(age),max(age)').from_(self.data).groupby('gender')
        actual = list(query.run())
        expected = [{'avg(age)': 19.0, 'gender': 'boy', 'max(age)': 20, 'min(age)': 18},
                    {'avg(age)': 17.0, 'gender': 'girl', 'max(age)': 18, 'min(age)': 16},
                    {'avg(age)': 56.0, 'gender': 'boy', 'max(age)': 56, 'min(age)': 56}]
        self.assertListEqual(actual, expected)

    def test_base_filter(self):
        query = select('age').from_(self.data).filter('gender=="boy"')
        actual = list(query.run())
        expected = [{'age': 18},
                    {'age': 20},
                    {'age': 56},
                   ]
        self.assertListEqual(actual, expected)

    def test_base_filter_groupby(self):
        query = select('min(age)').from_(self.data).groupby('gender').filter('age<50')
        actual = list(query.run())
        expected = [{'min(age)': 18},
                    {'min(age)': 16},
                   ]
        self.assertListEqual(actual, expected)

    def test_func(self):
        query = select('age+1').from_(self.data).filter('left(gender, 1)=="b"')
        actual = list(query.run())
        expected = [{'age+1': 19},
                    {'age+1': 21},
                    {'age+1': 57},
                   ]
        self.assertListEqual(actual, expected)

    def test_groupby_time(self):
        query = select('format_time(time, "1h"),min(age)').from_(self.data).groupby('format_time(time, "1h")')
        actual = list(query.run())
        expected = [{'format_time(time, "1h")': datetime(2023, 10, 1, 8, 0), 'min(age)': 16},
                    {'format_time(time, "1h")': datetime(2023, 10, 1, 9, 0), 'min(age)': 18},
                    {'format_time(time, "1h")': datetime(2023, 10, 1, 11, 0), 'min(age)': 56},
                   ]
        self.assertListEqual(actual, expected)

    def test_top_func(self):
        self.maxDiff = None
        query = select('format_time(time, "1h"),top(gender, 2)').from_(self.data).groupby('format_time(time, "1h")')
        actual = list(query.run())
        expected = [{'format_time(time, "1h")': datetime(2023, 10, 1, 8, 0), 'top(gender, 2)': [('boy', 2), ('girl', 1)]},
                    {'format_time(time, "1h")': datetime(2023, 10, 1, 9, 0), 'top(gender, 2)': [('girl', 1)]},
                    {'format_time(time, "1h")': datetime(2023, 10, 1, 11, 0), 'top(gender, 2)': [('boy', 1)]},
                   ]
        self.assertListEqual(actual, expected)
