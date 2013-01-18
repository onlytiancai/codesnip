#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "http_parser/http_parser.h"

static http_parser *parser; 

void
print_string(const char *message, const char *arg, size_t arg_len)
{
    char str[arg_len];
    strncpy(str, arg, arg_len);
    str[arg_len] = '\0';

    printf(message, str);
}

int
my_url_callback(http_parser *p, const char *buf, size_t len)
{
    print_string("url..\n\t%s\n", buf, len);
    return 0;
}

int
my_header_field_callback(http_parser *p, const char *buf, size_t len)
{
    print_string("header name..\n\t%s\n", buf, len);
    return 0;
}


int
my_header_value_callback(http_parser *p, const char *buf, size_t len)
{
    print_string("header value..\n\t%s\n", buf, len);
    return 0;
}

int
my_headers_complete_callback(http_parser *p)
{
    printf("headers complete..\n");
    return 0;
}

int
my_body_callback(http_parser *p, const char *buf, size_t len)
{
    print_string("body..\n\t%s\n", buf, len);
    return 0;
}


int
my_message_complete_callback(http_parser *p)
{
    printf("message complete ..\n");
    return 0;
}

int main(int argc, const char *argv[])
{
    size_t nparsed;

    http_parser_settings settings;
    settings.on_url = my_url_callback;
    settings.on_header_field = my_header_field_callback;
    settings.on_header_value = my_header_value_callback; 
    settings.on_headers_complete = my_headers_complete_callback;
    settings.on_body = my_body_callback; 
    settings.on_message_complete = my_message_complete_callback;

    parser = malloc(sizeof(http_parser));
    http_parser_init(parser, HTTP_REQUEST);

    char *buf = "POST /test HTTP/1.1\r\nHost: 0.0.0.0\r\n"
                "Accept: */*\r\n"
                "Content-Length: 12\r\n"
                "\r\n"
                "Hello world.";

    nparsed = http_parser_execute(parser, &settings, buf, strlen(buf));
    nparsed = http_parser_execute(parser, &settings, buf, 0);

    free(parser);
    return nparsed;
}
