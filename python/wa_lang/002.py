import unittest

def parse(s):
    return [ch for ch in s if ch != ' '] 

class MyTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(['1','+','2','*','3'], parse('1+2*3'))
        self.assertEqual(['1','+','2','*','3'], parse('1 + 2 * 3'))

if __name__ == "__main__":
    unittest.main()
