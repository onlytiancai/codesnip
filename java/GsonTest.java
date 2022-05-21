package com.ihuhao.demo;

import java.util.HashMap;
import java.util.Arrays;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public class GsonTest {
    public static void main(String args[]) {
        Gson gson = new GsonBuilder().setPrettyPrinting().create();

        HashMap<String, Object> map = new HashMap<>();
        map.put("name", "wawa");
        map.put("age", 18);
        map.put("skills", Arrays.asList("吃饭", "睡觉", "打豆豆"));

        String json  = gson.toJson(map);
        System.err.println(json);
    }
}
