#include <stdio.h>
#include <string.h>

const int MAXLEN = 100;

int main(int argc, char *argv[])
{
    int i, j, max_len = 0, max_l = 0;
    char* str1 = "1AB2345CD"; 
    char* str2 = "12345EF"; 
    char* expected = "2345";
    char actual[MAXLEN];
    int dp[MAXLEN][MAXLEN];

    printf("%s %s\n", str1, str2);

    for (i = 0; i < strlen(str1); ++i) {
       for (j = 0; j < strlen(str2); j++) {
           dp[i][j] = 0; 
           if (str1[i] == str2[j] && i > 0 && j > 0) {
               dp[i][j] = dp[i-1][j-1] + 1;
               if (dp[i][j] > max_len) {
                   max_len = dp[i][j];
                   max_l = i - max_len + 1;
               }
           }
       }
    }

    strncpy(actual, &str1[max_l], max_len);
    actual[max_len] = '\0';
    printf("%s %s\n", actual, expected);
    printf("%s\n", strcmp(expected, actual) == 0 ? 
            "\033[30;42mtest passed\033[0m": 
            "\033[41;32mtest failed\033[0m");

    return 0;
}
