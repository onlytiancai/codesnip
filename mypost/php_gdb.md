## How to get the call stack of a running PHP process

Sometimes our PHP service is hang, but we can't stop the service to debug it, it would be nice to live debug it.

We tried strace and XDebug, neither of them solved the problem. What should we do? Try the outdated GDB！

Prepare test code.

    $ cd /tmp
    $ vi test.php
    <?php
    function bar() {
        while (true) {
            sleep(1);
            echo date('c')."\n";
        }
    }

    function foo() {
        bar();
    }

    foo();

View PHP version.

    $ php8.1 --version
    PHP 8.1.7 (cli) (built: Jun 25 2022 08:12:59) (NTS)
    Copyright (c) The PHP Group
    Zend Engine v4.1.7, Copyright (c) Zend Technologies
        with Zend OPcache v8.1.7, Copyright (c), by Zend Technologies
        with Xdebug v3.1.2, Copyright (c) 2002-2021, by Derick Rethans

Run the script.

    $ php8.1 test.php
    2022-09-15T17:31:44+08:00
    2022-09-15T17:31:45+08:00
    2022-09-15T17:31:46+08:00
    2022-09-15T17:31:47+08:00

Find the PID.

    $ ps -ef | grep test.php
    ubuntu   3291225 3288744  0 17:31 pts/2    00:00:00 php8.1 test.php
    ubuntu   3291340 3273905  0 17:32 pts/3    00:00:00 grep test.php

Run GDB.

    $ sudo gdb -p 3291225
    GNU gdb (Ubuntu 9.2-0ubuntu1~20.04) 9.2
    Copyright (C) 2020 Free Software Foundation, Inc.

View C call stack, there is no useful information.

	(gdb) bt
	#0  0x00007fc69bba9334 in __GI___clock_nanosleep (clock_id=<optimized out>, clock_id@entry=0, flags=flags@entry=0,
		req=req@entry=0x7ffd831aa1b0, rem=rem@entry=0x7ffd831aa1b0) at ../sysdeps/unix/sysv/linux/clock_nanosleep.c:78
	#1  0x00007fc69bbaf047 in __GI___nanosleep (requested_time=requested_time@entry=0x7ffd831aa1b0,
		remaining=remaining@entry=0x7ffd831aa1b0) at nanosleep.c:27
	#2  0x00007fc69bbaef7e in __sleep (seconds=0) at ../sysdeps/posix/sleep.c:55
	#3  0x00005580fb520595 in ?? ()
	#4  0x00007fc699a2441d in xdebug_execute_internal (current_execute_data=0x7fc699814160, return_value=0x7ffd831aa2a0)
		at ./build-8.1/src/base/base.c:897
	#5  0x00005580fb65f554 in execute_ex ()
	#6  0x00007fc699a23b52 in xdebug_execute_ex (execute_data=0x7fc6998140e0) at ./build-8.1/src/base/base.c:779
	#7  0x00005580fb662255 in execute_ex ()
	#8  0x00007fc699a23b52 in xdebug_execute_ex (execute_data=0x7fc699814080) at ./build-8.1/src/base/base.c:779
	#9  0x00005580fb662255 in execute_ex ()
	#10 0x00007fc699a23b52 in xdebug_execute_ex (execute_data=0x7fc699814020) at ./build-8.1/src/base/base.c:779
	#11 0x00005580fb662a5d in zend_execute ()
	#12 0x00005580fb5f2f05 in zend_execute_scripts ()
	#13 0x00005580fb58ff2a in php_execute_script ()
	#14 0x00005580fb6db1ed in ?? ()
	#15 0x00005580fb4347a8 in ?? ()
	#16 0x00007fc69baf00b3 in __libc_start_main (main=0x5580fb4343a0, argc=2, argv=0x7ffd831adf08, init=<optimized out>,
		fini=<optimized out>, rtld_fini=<optimized out>, stack_end=0x7ffd831adef8) at ../csu/libc-start.c:308
	#17 0x00005580fb43494e in _start ()

View `executor_globals.current_execute_data`, looks like there is some useful information.

	(gdb) p executor_globals.current_execute_data
	$2 = (struct _zend_execute_data *) 0x7fc699814160
	(gdb) pt executor_globals.current_execute_data
	type = struct _zend_execute_data {
		const zend_op *opline;
		zend_execute_data *call;
		zval *return_value;
		zend_function *func;
		zval This;
		zend_execute_data *prev_execute_data;
		zend_array *symbol_table;
		void **run_time_cache;
		zend_array *extra_named_params;
	} *
	(gdb) set print pretty on
	(gdb) p *executor_globals.current_execute_data
	$3 = {
	  opline = 0x262220736920676e,
	  call = 0x747468203b0a2e22,
	  return_value = 0x7068702f2f3a7370,
	  func = 0x5580fc923670,
	  This = {
		value = {
		  lval = 0,
		  dval = 0,
		  counted = 0x0,
		  str = 0x0,
		  arr = 0x0,
		  obj = 0x0,
		  res = 0x0,
		  ref = 0x0,
		  ast = 0x0,
		  zv = 0x0,
		  ptr = 0x0,
		  ce = 0x0,
		  func = 0x0,
		  ww = {
			w1 = 0,
			w2 = 0
		  }
		},
		u1 = {
		  type_info = 0,
		  v = {
			type = 0 '\000',
			type_flags = 0 '\000',
			u = {
			  extra = 0
			}
		  }
		},
		u2 = {
		  next = 1,
		  cache_slot = 1,
		  opline_num = 1,
		  lineno = 1,
		  num_args = 1,
		  fe_pos = 1,
		  fe_iter_idx = 1,
		  property_guard = 1,
		  constant_flags = 1,
		  extra = 1
		}
	  },
	  prev_execute_data = 0x7fc6998140e0,
	  symbol_table = 0x72613b0a3a656c70,
	  run_time_cache = 0x6172617065735f67,
	  extra_named_params = 0x7074756f2e726f74
	}

View `function_name`, cool, it worked, we see the function on the call stack.

	(gdb) p (char*)executor_globals.current_execute_data.func.common.function_name.val
	$4 = 0x5580fc923658 "sleep"
	(gdb) p (char*)executor_globals.current_execute_data.prev_execute_data.func.common.function_name.val
	$5 = 0x7fc699863458 "bar"
	(gdb) p (char*)executor_globals.current_execute_data.prev_execute_data.prev_execute_data.func.common.function_name.val
	$6 = 0x7fc6998634f8 "foo"
	(gdb) p (char*)executor_globals.current_execute_data.prev_execute_data.prev_execute_data.prev_execute_data.func.common.function_name.val
	$7 = 0x18 <error: Cannot access memory at address 0x18>

Let's create a function to print more useful information.

	define phpbt
	  set $ed=executor_globals.current_execute_data
	  while $ed
		set $filename = "undefined"
		set $lineno = 0
		set $fun ="undefined"
		if $ed->func.op_array.opcodes
			set $filename = (char*)((zend_execute_data *)$ed)->func.op_array.filename.val        
		end
		
		if $ed->This.u1.type_info
			set $lineno = ((zend_execute_data *)$ed)->opline.lineno
		end
		
		if $ed->func.common.function_name
			set $fun = (char*)((zend_execute_data *)$ed)->func.common.function_name.val
		end
		printf "%s[%u]:%s\n", $filename, $lineno, $fun
		
		set $ed = ((zend_execute_data *)$ed)->prev_execute_data
	  end
	end

Okay, we can use it now.

	(gdb) phpbt
	undefined[0]:sleep
	/tmp/test.php[4]:bar
	/tmp/test.php[10]:foo
	/tmp/test.php[13]:undefined

Quit the debug session.

	(gdb) q
	A debugging session is active.
	Inferior 1 [process 3291225] will be detached.
	Quit anyway? (y or n) y
	Detaching from program: /usr/bin/php8.1, process 3291225

Automate everything

	$ vi phpbt.gdbscript
	bt
	define phpbt
	  set $ed=executor_globals.current_execute_data
	  while $ed
		set $filename = "undefined"
		set $lineno = 0
		set $fun ="undefined"
		if $ed->func.op_array.opcodes
			set $filename = (char*)((zend_execute_data *)$ed)->func.op_array.filename.val
		end

		if $ed->This.u1.type_info
			set $lineno = ((zend_execute_data *)$ed)->opline.lineno
		end

		if $ed->func.common.function_name
			set $fun = (char*)((zend_execute_data *)$ed)->func.common.function_name.val
		end
		printf "%s[%u]:%s\n", $filename, $lineno, $fun

		set $ed = ((zend_execute_data *)$ed)->prev_execute_data
	  end
	end
	phpbt

Run it.

	$ sudo gdb --batch -p 3291225 -x phpbt.gdbscript
	[Thread debugging using libthread_db enabled]
	Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
	0x00007fc69bba9334 in __GI___clock_nanosleep (clock_id=<optimized out>, clock_id@entry=0, flags=flags@entry=0, req=req@entry=0x7ffd831aa1b0, rem=rem@entry=0x7ffd831aa1b0) at ../sysdeps/unix/sysv/linux/clock_nanosleep.c:78
	78      ../sysdeps/unix/sysv/linux/clock_nanosleep.c: No such file or directory.
	#0  0x00007fc69bba9334 in __GI___clock_nanosleep (clock_id=<optimized out>, clock_id@entry=0, flags=flags@entry=0, req=req@entry=0x7ffd831aa1b0, rem=rem@entry=0x7ffd831aa1b0) at ../sysdeps/unix/sysv/linux/clock_nanosleep.c:78
	#1  0x00007fc69bbaf047 in __GI___nanosleep (requested_time=requested_time@entry=0x7ffd831aa1b0, remaining=remaining@entry=0x7ffd831aa1b0) at nanosleep.c:27
	#2  0x00007fc69bbaef7e in __sleep (seconds=0) at ../sysdeps/posix/sleep.c:55
	#3  0x00005580fb520595 in ?? ()
	#4  0x00007fc699a2441d in xdebug_execute_internal (current_execute_data=0x7fc699814160, return_value=0x7ffd831aa2a0) at ./build-8.1/src/base/base.c:897
	#5  0x00005580fb65f554 in execute_ex ()
	#6  0x00007fc699a23b52 in xdebug_execute_ex (execute_data=0x7fc6998140e0) at ./build-8.1/src/base/base.c:779
	#7  0x00005580fb662255 in execute_ex ()
	#8  0x00007fc699a23b52 in xdebug_execute_ex (execute_data=0x7fc699814080) at ./build-8.1/src/base/base.c:779
	#9  0x00005580fb662255 in execute_ex ()
	#10 0x00007fc699a23b52 in xdebug_execute_ex (execute_data=0x7fc699814020) at ./build-8.1/src/base/base.c:779
	#11 0x00005580fb662a5d in zend_execute ()
	#12 0x00005580fb5f2f05 in zend_execute_scripts ()
	#13 0x00005580fb58ff2a in php_execute_script ()
	#14 0x00005580fb6db1ed in ?? ()
	#15 0x00005580fb4347a8 in ?? ()
	#16 0x00007fc69baf00b3 in __libc_start_main (main=0x5580fb4343a0, argc=2, argv=0x7ffd831adf08, init=<optimized out>, fini=<optimized out>, rtld_fini=<optimized out>, stack_end=0x7ffd831adef8) at ../csu/libc-start.c:308
	#17 0x00005580fb43494e in _start ()
	undefined[0]:sleep
	/tmp/test.php[4]:bar
	/tmp/test.php[10]:foo
	/tmp/test.php[13]:undefined
	[Inferior 1 (process 3291225) detached]


Links:
- https://stackoverflow.com/questions/14261821/get-a-stack-trace-of-a-running-or-hung-php-script
- https://aurelien-riv.github.io/php/2019/12/07/which-function-php-executing.html
- https://stackoverflow.com/questions/12618331/displaying-struct-values-in-gdb
- https://github.com/php/php-src/blob/2a5dccd4beedb72c24fe95817e0687d7155be2ce/Zend/zend_compile.h
- https://stackoverflow.com/questions/10748501/what-are-the-best-ways-to-automate-a-gdb-debugging-session
