import unittest

def parse(s):
    return list(s)

class MyTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(['1','+','2','*','3'], parse('1+2*3'))

if __name__ == "__main__":
    unittest.main()
