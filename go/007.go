package main

import (
    "fmt"
    "time"
)

func main() {
    i := 2
    fmt.Print("Write", i, "is")
    switch i {
    case 1:
        fmt.Println("One")
    case 2:
        fmt.Println("Tow")
    case 3:
        fmt.Println("Three")
    }

    switch time.Now().Weekday() {
    case time.Saturday, time.Sunday:
        fmt.Println("It's the weekend")
    default:
        fmt.Println("It's the weekday")
    }
    
    t := time.Now()
    switch {
        case t.Hour() < 12:
            fmt.Println("It's before noon")
        default:
            fmt.Println("It's after noon")
    }

    whatAmI := func(i interface{}) {
        switch t := i.(type) {
        case bool:
            fmt.Println("I am bool")
        case int:
            fmt.Println("I am int")
        default:
            fmt.Printf("Don't know type %T\n", t)
        }
    }

    whatAmI(true)
    whatAmI(1)
    whatAmI("hey")

}
