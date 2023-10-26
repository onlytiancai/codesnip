from logsql import select 
import unittest

class SelectTest(unittest.TestCase):
    data = [{'gender': 'boy', 'age': 18},
            {'gender': 'boy', 'age': 20},
            {'gender': 'girl', 'age': 16},
            {'gender': 'girl', 'age': 18},
            {'gender': 'boy', 'age': 56},
           ] 

    def test_base_groupby(self):
        query = select('gender, avg(age), min(age), max(age)').from_(self.data).groupby('gender')
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
