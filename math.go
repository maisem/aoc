package aoc

import (
	"log"
	"math"
	"strconv"
	"strings"

	"golang.org/x/exp/constraints"
)

// Digits returns the individual digits of the string.
func Digits(line string) []int {
	var in []int
	for _, c := range line {
		in = append(in, Digit(c))
	}
	return in
}

// Digit returns the digit value of the rune.
func Digit(r rune) int {
	if r < '0' || r > '9' {
		log.Fatalf("not a digit: %q", r)
	}
	return int(r - '0')
}

// SolveQuad returns the roots of the quadratic equation ax^2 + bx + c = 0.
func SolveQuad[T Number](a, b, c T) (float64, float64) {
	d := float64(b*b - 4*a*c)
	if d < 0 {
		log.Fatalf("no real roots")
	}
	d = math.Sqrt(d)
	a2 := float64(2 * a)
	return (-float64(b) + d) / a2, (-float64(b) - d) / a2
}

// LCM returns the least common multiple of the integers.
func LCM(integers ...int) int {
	if len(integers) == 0 {
		panic("no integers")
	}
	if len(integers) == 1 {
		return integers[0]
	}

	lcm := func(a, b int) int {
		return a * b / GCD(a, b)
	}

	result := 1
	for i := 0; i < len(integers); i++ {
		result = lcm(result, integers[i])
	}

	return result
}

// GCD returns the greatest common divisor of the integers.
func GCD(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// Number is a type that can be used in math functions.
type Number interface {
	constraints.Float | constraints.Integer
}

// Sum returns the sum of the numbers.
func Sum[T Number](nums ...T) T {
	var sum T
	for _, v := range nums {
		sum += v
	}
	return sum
}

// Extrapolate returns the next value in the sequence x.
// If forward is true, it extrapolates the next value, otherwise
// it extrapolates the previous value in the sequence.
func Extrapolate[T Number](x []T, forward bool) (y T) {
	diffs := make([]T, 0, len(x))
	allZero := true
	for i := 1; i < len(x); i++ {
		d := x[i] - x[i-1]
		diffs = append(diffs, d)
		if d != 0 {
			allZero = false
		}
	}
	ix := 0
	if forward {
		ix = len(x) - 1
	}
	if allZero {
		return x[ix]
	}
	val := x[ix]
	diff := Extrapolate(diffs, forward)
	if forward {
		return val + diff
	}
	return val - diff
}

// ParseBinary parses a binary string.
func ParseBinary(in string) int64 {
	return MustGet(strconv.ParseInt(strings.TrimPrefix(in, "0b"), 2, 64))
}

// AbsDiff returns the absolute difference between x and y.
func AbsDiff[T Number](x, y T) T {
	v := x - y
	if v < 0 {
		v = -v
	}
	return v
}

// Int returns the int value of the string.
func Int(s string) int {
	return MustGet(strconv.Atoi(strings.TrimSpace(s)))
}

// Ints returns the int values of the strings.
func Ints(s ...string) []int {
	var out []int
	for _, v := range s {
		out = append(out, Int(v))
	}
	return out
}

// PolygonArea returns the area of the polygon defined by the points. It
// assumes the points are in clockwise order, and uses the shoelace formula.
func PolygonArea(pts []Pt) int {
	var area int

	for i := 1; i < len(pts); i++ {
		a := pts[i-1]
		b := pts[i]
		area += int(a.X*b.Y) - int(a.Y*b.X)
	}
	if area < 0 {
		area = -area
	}
	return area >> 1
}

// PolygonPerimeter returns the perimeter of the polygon defined by the points.
func PolygonPermimeter(pts []Pt) int {
	var perimeter int

	for i := 1; i < len(pts); i++ {
		a := pts[i-1]
		b := pts[i]
		perimeter += a.MDist(b)
	}
	return perimeter
}

// PolygonBoundedPoints returns the number of points with integer coordinates
// inside the polygon defined by the points.
func PolygonBoundedPoints(pts []Pt) int {
	/*
	  Pick's theorem:
	  A = i + b/2 - 1

	  Bounded points = i + b

	  i = A - b/2 + 1
	  i + b = A + b/2 + 1
	*/
	A := PolygonArea(pts)
	b_2 := PolygonPermimeter(pts) >> 1
	return A + b_2 + 1
}
