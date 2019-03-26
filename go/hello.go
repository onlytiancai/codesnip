package main

import "fmt"

func aaa (s *string) {
    s = nil
}

func bbb(s *string) {
    x :=  "hello"
    s = &x 
}

func main() {
    var s *string

    aaa(s) 
    if (s == nil) {
        fmt.Printf("s is nil\n")
    } else {
        fmt.Printf("%s\n", *s)
    }

    bbb(s)
    fmt.Printf("%s\n", *s)
}
