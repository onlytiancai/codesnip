#import <Foundation/Foundation.h>

int main(void) {
    // 用数字初始化字符串
    NSString *str = [[NSString alloc] initWithFormat:@"%d", 10]; 
    NSLog(@"1:%@!", str);
    [str release];

    // 用C字符串初始化字符串
    str = [[NSString alloc] initWithCString:"我是一个字符串"];
    NSLog(@"2:%@!", str);
    [str release];

    // 给一个字符串追加一个字符串
    NSMutableString *String1 = [[NSMutableString alloc] initWithString:@"This is a NSMutableString"];

    [String1 appendFormat:[NSString stringWithFormat:@", I will be adding some character"]];
    NSLog(@"3:%@!", String1);

    // 替换一个字符串中的字串
    [String1 replaceOccurrencesOfString:@"I" withString:@"You" options:NSLiteralSearch range:NSMakeRange(0, [String1 length])];
    NSLog(@"4:%@!", String1);

    // 在指定位置删除多个字符
    [String1 deleteCharactersInRange:NSMakeRange(0, 5)];
    NSLog(@"5:%@!", String1);

    // 在指定位置追加
    [String1 insertString:@"Hi! " atIndex:0];
    NSLog(@"6:%@!", String1);

    // 把制定范围内的字符替换成别的字符串
    [String1 replaceCharactersInRange:NSMakeRange(0, 4) withString:@"That"];
    NSLog(@"7:%@!", String1);

    // 判断字符串是否有某个前缀或后缀
    str = @"NSStringInformation.txt";
    [str hasPrefix:@"NSString"] == 1 ?  NSLog(@"YES") : NSLog(@"NO");
    [str hasSuffix:@".txt"] == 1 ?  NSLog(@"YES") : NSLog(@"NO");

    // 风格字符串
    str = @"Norman, Stanley, Fletcher";
    NSArray *listItems = [str componentsSeparatedByString:@", "];
    NSLog(@"8:%@!", listItems);

    // 是否包含该字符串
    NSRange range = [@"111222333" rangeOfString: @"222"];

    if (range.location == NSNotFound)
    {
        NSLog(@"found");
    }
    else
    {
        NSLog(@"not found");
    }

    [String1 release];

    return 0;
}
