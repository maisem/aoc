package aoc

import (
	"reflect"

	"golang.org/x/exp/constraints"
	"tailscale.com/util/deephash"
)

type Grid[T any] [][]T

func (g Grid[T]) At(p Pt) T {
	return g[p.Y][p.X]
}

func (g Grid[T]) Set(p Pt, v T) {
	g[p.Y][p.X] = v
}

func (g Grid[T]) AtOk(p Pt) (T, bool) {
	if p.X < 0 || p.Y < 0 || p.X >= len(g[0]) || p.Y >= len(g) {
		var zero T
		return zero, false
	}
	return g[p.Y][p.X], true
}

func MakeGrid[T any](x, y int) Grid[T] {
	out := make(Grid[T], y)
	for i := range out {
		out[i] = make([]T, x)
	}
	return out
}

type hashFn[T any] func(*T) deephash.Sum

var hashers map[reflect.Type]any // map[reflect.Type]hashFn[T]

func (g Grid[T]) Hash() deephash.Sum {
	if hashers == nil {
		hashers = make(map[reflect.Type]any)
	}
	rt := reflect.TypeOf(g)
	h, ok := hashers[rt]
	if !ok {
		h = deephash.HasherForType[Grid[T]]()
		hashers[rt] = h
	}
	return h.(func(*Grid[T]) deephash.Sum)(&g)
}

func (g Grid[T]) TransposeInto(out Grid[T]) {
	size := g.Size()
	for x := 0; x < size.X; x++ {
		for y := 0; y < size.Y; y++ {
			out[x][y] = g[y][x]
		}
	}
}

func (g Grid[T]) Transpose() Grid[T] {
	size := g.Size()
	out := MakeGrid[T](size.Y, size.X)
	g.TransposeInto(out)
	return out
}

func (g Grid[T]) RotateCounterClockwiseInto(out [][]T) {
	size := g.Size()
	for x := 0; x < size.X; x++ {
		for y := 0; y < size.Y; y++ {
			out[y][size.Y-1-x] = g[x][y]
		}
	}
}

func (g Grid[T]) RotateCounterClockwise() Grid[T] {
	size := g.Size()
	out := MakeGrid[T](size.Y, size.X)
	g.RotateCounterClockwiseInto(out)
	return out
}

func (g Grid[T]) Size() Pt {
	if len(g) == 0 {
		return Pt{}
	}
	return Pt{len(g[0]), len(g)}
}

func (g Grid[T]) EdgePaths() []Path {
	size := g.Size()
	var paths []Path
	for x := 0; x < size.X; x++ {
		paths = append(paths, Path{
			Pt:  Pt{x, 0},
			Dir: Down,
		}, Path{
			Pt:  Pt{x, size.Y - 1},
			Dir: Up,
		})
	}
	for y := 0; y < size.Y; y++ {
		paths = append(paths, Path{
			Pt:  Pt{0, y},
			Dir: Right,
		}, Path{
			Pt:  Pt{size.X - 1, y},
			Dir: Left,
		})
	}
	return paths
}

// ToGraph converts the grid into a graph. If allowDiagonals is true, then
// diagonal neighbors are included. If disallowed is not nil, it is additionally
// called on each cell, and if it returns true, that cell is not included in the
// graph.
func (grid Grid[T]) ToGraph(start Pt, allowDiagonals bool, disallowed func(T) bool) Graph[Pt] {
	var g Graph[Pt]
	g.Nodes = make(map[Pt]bool)
	g.Edges = make(map[Pt]map[Pt]int)

	fn := Pt.ForImmediateNeighbors
	if allowDiagonals {
		fn = Pt.ForNeighbors
	}

	q := NewQueue[Pt](start)
	q.While(func(p1 Pt) bool {
		if _, ok := g.Nodes[p1]; ok {
			return true
		}
		g.Nodes[p1] = true
		fn(p1, func(p2 Pt) (keepGoing bool) {
			if v, ok := grid.AtOk(p2); !ok || disallowed(v) {
				return true
			}
			if _, ok := g.Nodes[p2]; ok {
				return true // already visited
			}
			q.Push(p2)
			if g.Edges[p2] == nil {
				g.Edges[p2] = make(map[Pt]int)
			}
			if g.Edges[p1] == nil {
				g.Edges[p1] = make(map[Pt]int)
			}
			g.Edges[p1][p2] = 1
			g.Edges[p2][p1] = 1
			return true
		})
		return true
	})
	g.Collapse()
	return g
}

// Path is a point and a direction.
type Path struct {
	Pt  Pt
	Dir Direction
}

func (g Grid[T]) Move(p Path) (Path, bool) {
	switch p.Dir {
	case 0:
		p.Pt.Y--
	case 1:
		p.Pt.X++
	case 2:
		p.Pt.Y++
	case 3:
		p.Pt.X--
	}
	size := g.Size()
	if p.Pt.X < 0 || p.Pt.Y < 0 || p.Pt.X >= size.X || p.Pt.Y >= size.Y {
		return Path{}, false
	}
	return p, true
}

// FloodFill fills all empty cells reachable from start with fill.
func FloodFill[T comparable](grid Grid[T], start Pt, empty, fill T) int {
	var q Queue[Pt]
	q.Push(start)
	for p, ok := q.Pop(); ok; p, ok = q.Pop() {
		p.ForNeighbors(func(p Pt) (keepGoing bool) {
			if grid.At(p) == empty {
				q.Push(p)
				grid[p.Y][p.X] = fill
			}
			return true
		})
	}
	return 0
}

type Direction int

const (
	Up Direction = iota
	Right
	Down
	Left
)

func (d Direction) Turn(right bool) Direction {
	switch d {
	case Up:
		if right {
			return Right
		}
		return Left
	case Right:
		if right {
			return Down
		}
		return Up
	case Down:
		if right {
			return Left
		}
		return Right
	case Left:
		if right {
			return Up
		}
		return Down
	}
	panic("bad")
}

func (d Direction) String() string {
	switch d {
	case Left:
		return "<"
	case Right:
		return ">"
	case Up:
		return "^"
	case Down:
		return "v"
	}
	return ""
}

// Segment is a line segment between two points.
type Segment struct {
	A, B Pt
}

type Pt = Pt2[int]

type Pt2[T constraints.Signed] struct {
	X, Y T
}

func (p Pt2[T]) ForImmediateNeighbors(f func(Pt2[T]) (keepGoing bool)) {
	p.ForNeighbors(func(n Pt2[T]) bool {
		if p.X == n.X || p.Y == n.Y {
			return f(n)
		}
		return true
	})
}

func (p Pt2[T]) ForNeighbors(f func(Pt2[T]) (keepGoing bool)) {
	for y := T(-1); y <= 1; y++ {
		for x := T(-1); x <= 1; x++ {
			if x == 0 && y == 0 {
				continue
			}
			if !f(Pt2[T]{p.X + x, p.Y + y}) {
				return
			}
		}
	}
}

func StandardizePt(p, size Pt) Pt {
	if p.X < 0 || p.Y < 0 || p.X >= size.X || p.Y >= size.Y {
		p.X = p.X % size.X
		p.Y = p.Y % size.Y
		if p.X < 0 {
			p.X += size.X
		}
		if p.Y < 0 {
			p.Y += size.Y
		}
	}
	return p
}

// MDist returns the manhattan distance between a and b.
func (a Pt2[T]) MDist(b Pt2[T]) T {
	return AbsDiff[T](a.X, b.X) + AbsDiff[T](a.Y, b.Y)
}

// Toward returns a point moving from p to b in max 1 step in the X
// and/or Y direction.
func (p Pt2[T]) Toward(b Pt2[T]) Pt2[T] {
	p1 := p
	if b.X < p.X {
		p1.X--
	} else if b.X > p.X {
		p1.X++
	}
	if b.Y < p.Y {
		p1.Y--
	} else if b.Y > p.Y {
		p1.Y++
	}
	return p1
}
