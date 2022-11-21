package main

import (
   "C"
)

//export echo
func echo(namePtr *C.char) *C.char {
   name := C.GoString(namePtr)
   return C.CString(name)
}

func main() {
}
