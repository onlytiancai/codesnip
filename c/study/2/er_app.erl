%% http://stackoverflow.com/questions/2206933/how-to-write-a-simple-webserver-in-erlang
-module(er_app).
-export([start/0]).

start() ->
    spawn(fun () -> {ok, Sock} = gen_tcp:listen(7000, [{active, false}]), 
                    loop(Sock) end).

loop(Sock) ->
    {ok, Conn} = gen_tcp:accept(Sock),
    Handler = spawn(fun () -> handle(Conn) end),
    gen_tcp:controlling_process(Conn, Handler),
    loop(Sock).

handle(Conn) ->
    ok = gen_tcp:send(Conn, response("Hello World")),
    ok = gen_tcp:close(Conn).

response(Str) ->
    B = iolist_to_binary(Str),
    iolist_to_binary(
      io_lib:fwrite(
         "HTTP/1.1 200 OK\nContent-Type: text/html\nContent-Length: ~p\n\n~s",
         [size(B), B])).
