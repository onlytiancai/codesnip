import unittest

def parse(s):
    ret = []
    temp = ''
    in_number = False
    for ch in s:
        if '0' <= ch <= '9':
            temp += ch
            if not in_number:
                in_number = True
        else:
            in_number = False 
            if temp != '':
                ret.append(temp)
                temp = ''
            if ch != ' ':
                ret.append(ch)
    if temp != '':
        ret.append(temp)
    return ret


class MyTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(['1','+','2','*','3'], parse('1+2*3'))
        self.assertEqual(['1','+','2','*','3'], parse('1 + 2 * 3'))
        self.assertEqual(['11','+','22','*','33'], parse('11 + 22 * 33'))

if __name__ == "__main__":
    unittest.main()
