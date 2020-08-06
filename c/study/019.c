#include <assert.h>

int myfunc(int i) {
    i = i + 2;
    i = i * 2;
    return i;
}

int main(void) {
    assert(myfunc(1) == 6);
    assert(myfunc(2) == 8);
    return 0;
}
