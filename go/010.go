package main

import (
    "fmt"
    "sort"
) 

func main() {
    strs := []string{"c", "a", "b"}
    sort.Strings(strs)
    fmt.Println("Strings", strs)

    ints := []int{3, 1, 2}
    sort.Ints(ints)
    fmt.Println("Ints", ints)

    s := sort.IntsAreSorted(ints)
    fmt.Println("Sorted", s)
}
