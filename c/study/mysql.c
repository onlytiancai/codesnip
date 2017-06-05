//   gcc mysql.c `mysql_config --cflags --libs`
#include <stdio.h>
#include <stdlib.h>
#include <mysql/my_global.h>
#include <mysql/mysql.h>

int main(int argc, char **argv) 
{
    printf("MySQL client version: %s\n", mysql_get_client_info());

    MYSQL *conn;
    MYSQL_RES *result;
    MYSQL_ROW row;
    MYSQL_FIELD *field;
    int num_fields;
    int i;

    conn = mysql_init(NULL);
    if (conn == NULL) {
        printf("Error: %u %s\n", mysql_errno(conn), mysql_error(conn));
        return 1; 
    }

    if (mysql_real_connect(conn, "localhost", "root", "", "mysql", 3306, NULL, 0) == NULL) {
        printf("Error: %u %s\n", mysql_errno(conn), mysql_error(conn));
        return 1; 
    }

    if (mysql_query(conn, "select user, host from user")) {
        printf("Error: %u %s\n", mysql_errno(conn), mysql_error(conn));
        return 1; 
    }

    result = mysql_store_result(conn);
    num_fields = mysql_num_fields(result);
    while ((row = mysql_fetch_row(result))) {
        for (i = 0; i < num_fields; i++) {
            if (i == 0) {
                while ((field = mysql_fetch_field(result))) {
                    printf("%s ", field->name);
                }
                printf("\n");
            }
            printf("%s ", row[i] ? row[i]: "NULL"); 
        }
        printf("\n");
    }

    mysql_free_result(result);
    mysql_close(conn);

    return 0;
}
