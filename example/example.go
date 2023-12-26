package main

import (
	_ "embed"

	"github.com/maisem/aoc"
)

func main() {
	aoc.Run(2023, source, &solver{})
}

//go:embed example.go
var source []byte

type solver struct {
	*aoc.Puzzle
}

/*
want=142

1abc2
pqr3stu8vwx
a1b2c3d4e5f
treb7uchet
*/
func (s solver) D1p1() any {
	s.ForLines(func(line string) {
		// do something
	})
	return "not-implemented"
}

// want=???
func (s solver) D1p2() any {
	return "not-implemented"
}
