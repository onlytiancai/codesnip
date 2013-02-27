#import <Foundation/Foundation.h>

@interface Cat: NSObject{
    int total;
}
- (void) eat: (NSString*) food;
@end

@implementation Cat
- (id) init {
    self = [super init];
    if (self) {
        total = 0;
    }
    return self;
}
- (void) eat: (NSString*) food{
    total += 1;
    NSLog(@"eating %d %@", total, food);
}
@end

int main(void) {
    Cat *cat = [[Cat alloc] init];
    [cat eat: @"fish"];
    [cat eat: @"checking"];
    [cat release];
    return 0;
}
