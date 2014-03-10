/* 编译
 * gcc -o run3 main3.c ./libuv/libuv.a ./http_parser/http_parser.o -lpthread -lrt
 *
 * 性能测试
 * ab -n 100000 -c 10 http://localhost:7000/
 *
 * 本机测试结果如下
 *
 * Concurrency Level:      10
 * Time taken for tests:   2.728 seconds
 * Complete requests:      100000
 * Failed requests:        0
 * Write errors:           0
 * Total transferred:      7900000 bytes
 * HTML transferred:       1400000 bytes
 * Requests per second:    36658.45 [#/sec] (mean)
 * Time per request:       0.273 [ms] (mean)
 * Time per request:       0.027 [ms] (mean, across all concurrent requests)
 * Transfer rate:          2828.14 [Kbytes/sec] received
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "./libuv/include/uv.h"
#include "./http_parser/http_parser.h"

//#define LOGF(fmt, params...) printf(fmt "\n", params);
#define LOGF(fmt, params...) ; // INFO: 为测性能，临时关闭日志输出

static int request_num = 1;
static uv_loop_t *loop;
static http_parser_settings settings;

typedef struct{
    uv_tcp_t handle;
    http_parser parser;
    uv_write_t write_req;
    int request_num;
} client_t;



uv_buf_t
alloc_buffer(uv_handle_t *handle, size_t suggested_size) {
    return uv_buf_init((char*) malloc(suggested_size), suggested_size);
}


void 
on_close(uv_handle_t* handle) {
    client_t* client = (client_t*) handle->data;
    LOGF("[ %5d ] connection closed", client->request_num);
    
    uv_write_t req = client -> write_req;
    char *base = (char*) req.data;
    free(base);
    free(client);
}

void
after_write(uv_write_t *req, int status) {
    if (status == -1) {
        fprintf(stderr, "Write error %s\n", uv_err_name(uv_last_error(loop)));
    }
    uv_close((uv_handle_t*)req->handle, on_close);
}

void
echo_read(uv_stream_t *tcp, ssize_t nread, uv_buf_t buf) {
    if (nread == -1) {
        if (uv_last_error(loop).code != UV_EOF)
            fprintf(stderr, "Read error %s\n", uv_err_name(uv_last_error(loop)));
        uv_close((uv_handle_t*) tcp, on_close);
        return;
    }

    size_t parsed;
    client_t* client = (client_t*) tcp->data;
    LOGF("[ %5d ] on read\n", client->request_num);

    parsed = http_parser_execute(&client->parser, &settings, buf.base, nread);
    if (parsed < nread) {
        printf("parse error\n");
        uv_close((uv_handle_t*) &client->handle, on_close);
    }
}

void
on_new_connection(uv_stream_t *server, int status) {
    if (status == -1) {
        // error!
        return;
    }

    client_t* client = (client_t*)malloc(sizeof(client_t));
    client->request_num = request_num;
    request_num++;

    LOGF("[ %5d ] new connection\n", request_num);

    uv_tcp_init(loop, &client->handle);
    http_parser_init(&client->parser, HTTP_REQUEST);

    client->parser.data = client;
    client->handle.data = client;

    if (uv_accept(server, (uv_stream_t*) &client->handle) == 0) {
        uv_read_start((uv_stream_t*) &client->handle, alloc_buffer, echo_read);
    }
    else {
        uv_close((uv_handle_t*) &client->handle, on_close);
    }
}

int
my_message_complete_callback(http_parser *parser)
{
    client_t* client = (client_t*) parser->data;
    char *rep = "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/plain\r\n"
                "Content-Length: 14\r\n"
                "\r\n"
                "Hello world.\r\n";

    uv_buf_t resbuf;
    resbuf.base = rep;
    resbuf.len = strlen(rep);

    int r = uv_write(&client->write_req, (uv_stream_t*)&client->handle, &resbuf, 1, after_write);

    return 0;
}

int
main() {
    settings.on_message_complete = my_message_complete_callback;
    loop = uv_default_loop();

    uv_tcp_t server;
    uv_tcp_init(loop, &server);

    struct sockaddr_in bind_addr = uv_ip4_addr("0.0.0.0", 7000);
    uv_tcp_bind(&server, bind_addr);
    int r = uv_listen((uv_stream_t*) &server, 128, on_new_connection);
    if (r) {
        fprintf(stderr, "Listen error %s\n", uv_err_name(uv_last_error(loop)));
        return 1;
    }
    return uv_run(loop, UV_RUN_DEFAULT);
}
