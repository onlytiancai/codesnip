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
    int i;
    char * temp = NULL;

    arr -> length = 0;
    for (i = 0; i < MAX_ARRAY_LENGTH; i++) {
        temp = (char*)malloc(MAX_STRING_LENGTH + 1);
        if (NULL == temp) {
            printf("malloc error\n");
            exit(1);
        }
        memset(temp, 0, MAX_STRING_LENGTH + 1);
        arr -> array[i] = temp; 
    }
}

void
destory_string_array(struct string_array * arr){
    int i;

    for (i = 0; i < MAX_ARRAY_LENGTH; i++) {
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
    int scan_index = 0;
    size_t len_input = strlen(input), current_len = 0, copy_length;
    char ch, *p = input, *dst;

    while ((ch = *(input++)) != '\0') {
        scan_index++;
        current_len++;
        if (' ' == ch || scan_index == len_input) {
            if (MAX_ARRAY_LENGTH == arr -> length) {
                printf("array length over flow\n");
                exit(1);
            }
            dst = arr -> array[(arr -> length)++];

            copy_length = scan_index == len_input ? current_len : current_len - 1;
            if (copy_length > MAX_STRING_LENGTH)
                copy_length = MAX_ARRAY_LENGTH;

            snprintf(dst, copy_length, "%s", p);

            printf("debug1:scan_index=%d arr.length=%ld dst=%s p=%s\n", scan_index, arr -> length, dst, p);

            current_len = 0;
            p = input;
        }
    }
}

int
main(int argc, char ** argv) {
    struct string_array arr;

    if (argc !=2){
        printf("Usage: split_by_blank \"POST / HTTP/1.0\"\n");
        return 1;
    }

    printf("%s\n", argv[1]);

    init_string_array(&arr);

    split_by_blank(argv[1], &arr);
    print_string_array(&arr);

    destory_string_array(&arr);

    return 0;
}
