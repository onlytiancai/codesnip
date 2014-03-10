#import <Foundation/Foundation.h>

@interface Cat: NSObject
  - (void) eat: (NSString*) food;
@end

@implementation Cat
- (void) eat: (NSString*) food{
    NSLog(@"eating %@", food);
}
@end

int main(void) {
    Cat *cat = [[Cat alloc] init];
    [cat eat: @"fish"];
    [cat release];
    return 0;
}
