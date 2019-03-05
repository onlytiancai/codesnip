using System;
using System.Collections.Generic;

namespace myApp
{
    class Task<T> { public T MyT {get; set;} }

    class Program  {
        static void Main(string[] args) {
            Task<IList<int>> t = Get<IList<int>>();       
            Console.WriteLine(t.MyT.Count);           
            Console.WriteLine("ok");           
        }

        static Task<T> Get<T>() {
            var ret =  new Task<T>();            
            Type itemType = typeof(T).GenericTypeArguments[0];
            var constructedListType = typeof(List<>).MakeGenericType(itemType);
            var instance = Activator.CreateInstance(constructedListType);
            ret.MyT = (T)instance;
            return ret;
        }        
    }
}