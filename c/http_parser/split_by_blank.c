// gcc split_by_blank.c -o split_by_blank.o 
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_ARRAY_LENGTH  10
#define MAX_STRING_LENGTH  10

struct string_array {
    size_t length;
    char *array[MAX_ARRAY_LENGTH];
};

void
init_string_array(struct string_array * arr) {
    arr -> length = 0;
    int i;
    for (i = 0; i < MAX_ARRAY_LENGTH; i++) {
        char * temp = (char*)malloc(MAX_STRING_LENGTH + 1);
        memset(temp, 0, MAX_STRING_LENGTH + 1);
        arr -> array[i] = temp; 
    }
}

void
destory_string_array(struct string_array * arr){
    int i;
    for (i = 0; i < arr -> length; i++) {
        free(arr -> array[i]);
    }
}

void
print_string_array(struct string_array * arr){
    int i;
    for (i = 0; i < arr -> length; i++) {
        printf("%d-%s\n", i, arr -> array[i]);
    }
}

void
split_by_blank(char * input, struct string_array * arr) {
    int scan_index = 0, current_len = 0;
    size_t len_input = strlen(input);
    char ch, *p = input;

    while ((ch = *(input++)) != '\0') {
        scan_index++;
        current_len++;
        if (ch == ' ' || scan_index == len_input) {
            if (arr -> length == MAX_ARRAY_LENGTH) {
                printf("array length over flow\n");
                exit(1);
            }
            char * dst = arr -> array[(arr -> length)++];

            size_t copy_length = scan_index == len_input ? current_len : current_len - 1;
            if (copy_length > MAX_STRING_LENGTH)
                copy_length = MAX_ARRAY_LENGTH;

            strncpy(dst, p, copy_length);

            printf("debug1:scan_index=%d arr.length=%ld dst=%s p=%s\n", scan_index, arr -> length, dst, p);

            current_len = 0;
            p = input;
        }
    }
}

int
main(int argc, char ** argv) {
    if (argc !=2){
        printf("Usage: split_by_blank \"POST / HTTP/1.0\"\n");
        return 1;
    }

    printf("%s\n", argv[1]);

    struct string_array arr;
    init_string_array(&arr);

    split_by_blank(argv[1], &arr);
    print_string_array(&arr);

    destory_string_array(&arr);

    return 0;
}
