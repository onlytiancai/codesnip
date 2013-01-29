package main
import (
    "fmt"
    "net/http"
    )

func handler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "hello world.")
}

func main() {
    http.HandleFunc("/", handler)
        fmt.Printf("Starting server")
        http.ListenAndServe(":7000", nil)
}
