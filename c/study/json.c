/* echo "/usr/local/lib" >> /etc/ld.so.conf
 * sudo /sbin/ldconfig -v | grep json
 * gcc json.c -I /usr/local/include/cjson/ -L /usr/local/lib/  -lcjson
 * */ 
#include <stdio.h>
#include <cJSON.h>
char * my_json_string = "{ \"name\": \"Jack (\\\"Bee\\\") Nimble\", \"format\": { \"type\": \"rect\", \"width\": 1920, \"height\": 1080, \"interlace\": false, \"frame rate\": 24 } } ";

int main(int argc, char **argv)
{

    printf("%s \n", my_json_string);

    cJSON * root = cJSON_Parse(my_json_string);
    cJSON * format = cJSON_GetObjectItemCaseSensitive(root, "format");
    cJSON * framerate_item = cJSON_GetObjectItemCaseSensitive(format, "frame rate");
    double framerate = 0;
    if (cJSON_IsNumber(framerate_item)) {
        framerate = framerate_item->valuedouble; 
        printf("framerate: %f \n", framerate);
    }

    cJSON_SetNumberValue(framerate_item, 25);
    char * rendered = cJSON_Print(root);
    printf("%s\n", rendered);

    cJSON_Delete(root);
    return 0;
}

