#encoding=utf-8
'''
找出一个锯齿数组里长度大于5的子数组
在符合要求的子数组里的数据里找出所有偶数
如果数据小于10的话乘以2,大于10的除以2
最后统计符合要求的数据的和
'''
inputdata = [
       [2,8,9,13,72,67,88,35,44],
       [33,28,47,2,10,45,66,92],
       [22,34,60,43,0,72,52],
       [10,11,53,58]
        ]

def sum1(input):
    return sum((lambda x: x < 10 and x*2 or x/2)(num)
            for seq in inputdata 
            if len(seq) >= 5
            for num in seq
            if num % 2 == 0)

def sum2(input):
    def getsublist():
        for sublist in input:
            if len(sublist) >= 5:
                yield sublist
    def filterdata(sublist):
        for data in sublist:
            if data % 2 == 0:
                yield data
    def processdata(data):
        if data < 10:
            return data * 2 
        return data / 2
    result = 0
    for sublist in getsublist():
        for data in filterdata(sublist):
            result += processdata(data)
    return result

if __name__ == '__main__':
    print sum1(inputdata)
    print sum2(inputdata)
