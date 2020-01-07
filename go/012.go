package main

import "os"

func main() {
    panic("a problem")

    os.Create("/tmp/file")
}
