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

elf

    readelf -a test.o
    readelf -w test.o

    >>> t = [die for die in dies if 'DW_AT_name' in die.attributes and die.attributes['DW_AT_name'].value == b'AAA']
    >>> t = t[0]
    >>> sec = [sec for sec in sections if sec.name == '.debug_info'][0]
    >>> t = [die for die in dies if 'DW_AT_name' in die.attributes and die.attributes['DW_AT_name'].value == b'int'][0]
    >>>> t = [die for die in dies if 'DW_AT_name' in die.attributes and die.attributes['DW_AT_name'].value == b'int'][0]
>>> t.offset
101
>> t.offset
    101

nginx

    ./configure --with-debug --with-cc-opt='-O0 -g' ...
    readelf -w ./objs/nginx | grep ngx_http_request_s
    python3 dump_struct.py /home/ubuntu/download/nginx-1.19.10/objs/nginx ngx_http_request_s

    >>> dies_offset_map[827642].cu.cu_offset
    825059
    >>> ch = list(dies_name_map[b'ngx_http_request_s'].iter_children())[0]
    >>> ch.attributes['DW_AT_type'].value
    2583
    >>> 825059+2583
    827642

    ch.get_DIE_from_attribute('DW_AT_type').get_DIE_from_attribute('DW_AT_type').get_DIE_from_attribute('DW_AT_type')

    >>> set([ch.get_DIE_from_attribute('DW_AT_type').tag for ch in chs])
    {'DW_TAG_pointer_type', 'DW_TAG_base_type', 'DW_TAG_array_type', 'DW_TAG_typedef'}

    vi ~/download/nginx-1.19.10/src/http/ngx_http_request.h

    >>> from elftools.elf.elffile import ELFFile
    >>> elf_info = ELFFile(open('/home/ubuntu/download/nginx-1.19.10/objs/nginx','rb'))
    >>> dwarf_info = elf_info.get_dwarf_info()

    def get_die(struct_name):
        for i, cu in enumerate(dwarf_info.iter_CUs()):
            for die in cu.iter_DIEs():
                if 'DW_AT_name' in die.attributes and die.attributes['DW_AT_name'].value == struct_name:
                    return die
    die = get_die(b'ngx_http_request_s')
    chs = list(die.iter_children())
    >>> chs[0].attributes['DW_AT_name'].value
    b'signature'
    >>> chs[0].get_DIE_from_attribute('DW_AT_type').tag
    'DW_TAG_typedef'
    >>> chs[0].get_DIE_from_attribute('DW_AT_type').attributes['DW_AT_name'].value
    b'uint32_t'
    chs[0].get_DIE_from_attribute('DW_AT_type').get_DIE_from_attribute('DW_AT_type').tag
    >>> chs[0].get_DIE_from_attribute('DW_AT_type').get_DIE_from_attribute('DW_AT_type').attributes['DW_AT_name'].value
    b'__uint32_t'
    >>> chs[0].get_DIE_from_attribute('DW_AT_type').get_DIE_from_attribute('DW_AT_type').get_DIE_from_attribute('DW_AT_type').tag
    'DW_TAG_base_type'
    >>> chs[0].get_DIE_from_attribute('DW_AT_type').get_DIE_from_attribute('DW_AT_type').get_DIE_from_attribute('DW_AT_type').attributes['DW_AT_name'].value
    b'unsigned int'

    >>> chs[1].tag
    'DW_TAG_member'
    >>> chs[1].attributes['DW_AT_name'].value
    b'connection'
    >>> chs[1].get_DIE_from_attribute('DW_AT_type').tag
    'DW_TAG_pointer_type'
    >>> chs[1].get_DIE_from_attribute('DW_AT_type').get_DIE_from_attribute('DW_AT_type').tag
    'DW_TAG_typedef'
    >>> chs[1].get_DIE_from_attribute('DW_AT_type').get_DIE_from_attribute('DW_AT_type').attributes['DW_AT_name'].value
b'ngx_connection_t'
    >>> chs[1].get_DIE_from_attribute('DW_AT_type').get_DIE_from_attribute('DW_AT_type').get_DIE_from_attribute('DW_AT_type').tag
    'DW_TAG_structure_type'

>>> chs[2].tag
'DW_TAG_member'
>>> chs[2].attributes['DW_AT_name'].value
b'ctx'
>>> chs[2].get_DIE_from_attribute('DW_AT_type').tag
'DW_TAG_pointer_type'
>>> chs[2].get_DIE_from_attribute('DW_AT_type').get_DIE_from_attribute('DW_AT_type').tag
'DW_TAG_pointer_type'


    typedef void (*ngx_http_event_handler_pt)(ngx_http_request_t *r);
