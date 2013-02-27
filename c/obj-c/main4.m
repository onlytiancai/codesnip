#import <Foundation/Foundation.h>

@protocol Eat
  - (void) eat: (NSString*) food;
@end

@interface Cat: NSObject <Eat>
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
