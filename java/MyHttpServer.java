import java.util.HashMap;
import java.util.Map;
import java.io.IOException;
import java.net.InetSocketAddress;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

@FunctionalInterface
interface MyHandlerInterface{
    void handle(MyContext ctx);
}

class MyContext {
    HttpExchange ex;
    public MyContext(HttpExchange ex) {
        this.ex = ex; 
    }

    public String getQuery(String name) {
        String queryParam = this.ex.getRequestURI().getQuery();
        String[] paramArray = queryParam.split("&");
        for (String param : paramArray) {
            String[] keyValue = param.split("=");
            String key = keyValue[0];
            String value = keyValue[1];
            if (key.equals(name)) return value;
        }
        return "";
    }

    public void send(int code, String out) {
        try{
            this.ex.sendResponseHeaders(200, out.length());
            this.ex.getResponseBody().write(out.getBytes());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}

class MiniSpringWeb {
    int port;
    Map<String, MyHandlerInterface> router = new HashMap<String, MyHandlerInterface>();

    public MiniSpringWeb(int port) {
        this.port = port;
    } 

    public void start () {
        try {
            HttpServer httpServer = HttpServer.create(new InetSocketAddress(port), 0);
            System.out.println("server start ...");

            for(String path: this.router.keySet()) {
                MyHandlerInterface handler = this.router.get(path);
                System.out.printf("add route %s %s\n", path, handler);
                httpServer.createContext(path, new HttpHandler() {
                    @Override
                    public void handle(HttpExchange ex) throws IOException {
                        handler.handle(new MyContext(ex));
                    }
                });
            }
            httpServer.start();
        } catch (Exception e) {
            e.printStackTrace();
        }
    } 

    public void handle(String path, MyHandlerInterface handler) {
        this.router.put(path, handler);
    }
}


public class MyHttpServer {
    public static void main(String[] args) {
        MiniSpringWeb server = new MiniSpringWeb(8080);

        server.handle("/", ctx -> {
            ctx.send(200, String.format("hello %s", ctx.getQuery("name")));
        }); 

        server.start();
    }
}
