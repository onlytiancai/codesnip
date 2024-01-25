使用
    gcc test.c -o test.o
    objdump -tT test.o 
    sudo bpftrace -e 'uprobe:./test.o:foo{printf("begin foo: arg 0=%d\n", arg0);@start = nsecs;}uretprobe:./test.o:foo{printf("end foo: retval=%d, cost=%ld ns\n", retval, nsecs-@start);}END{printf("program end\n");clear(@start);}' 
    readelf --debug-dump=info test.o | grep foo


nginx

    which nginx
    objdump -tT /usr/sbin/nginx
    bpftrace -e 'uprobe:/usr/sbin/nginx:ngx_http_process_request_uri{printf("process url:%d\n", arg0)}'
    find ~/src/nginx/src/ -type f | xargs grep ngx_http_process_request_uri
        src/http/ngx_http.h:ngx_int_t ngx_http_process_request_uri(ngx_http_request_t *r);

    /home/ubuntu/src/nginx/src/http/ngx_http_request.h
    sudo bpftrace -I /home/ubuntu/src/nginx/src/ nginx.bt
    readelf -s `which nginx`
    file `which nginx`
    readelf -s `which nginx` | fgrep 'Symbol table'
    nginx-debug -V 2>&1 | grep -- '--with-debug'

example

    struct nameidata {
            struct path     path;
            struct qstr     last;
            // [...]
    };

    printf("open path: %s\n", str(((struct path *)arg0)->dentry->d_name.name));

    sudo apt-get install gcc-multilib
