//go:build e2e

package main

// The automation plugin lets another process drive this app: read its
// windows and elements, type into them, press its buttons and capture it
// as a picture. It is only in builds made with -tags e2e, so it never
// ships.
//
//	go build -tags e2e -o /tmp/mygo-e2e .
//	MYGO_AUTOMATION=/tmp/mygo-e2e.sock /tmp/mygo-e2e
import _ "github.com/egoist/mygo/plugins/automation"
