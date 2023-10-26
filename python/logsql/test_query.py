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
        query = select('avg(age), min(age), max(age)').from_(self.data).groupby('gender')
        actual = list(query.run())
        expected = [{'avg(age)': 19.0, 'gender': 'boy', 'max(age)': 20, 'min(age)': 18},
                    {'avg(age)': 17.0, 'gender': 'girl', 'max(age)': 18, 'min(age)': 16},
                    {'avg(age)': 56.0, 'gender': 'boy', 'max(age)': 56, 'min(age)': 56}]
        self.assertListEqual(actual, expected)
