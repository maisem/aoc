// Package aoc are quick & dirty utilities for helping Maisem
// solve Advent of Code problems. (forked from bradfitz/aoc)
package aoc

import (
	"bufio"
	"bytes"
	"container/heap"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/exp/constraints"
	"golang.org/x/exp/maps"
	"tailscale.com/util/deephash"
)

type sample struct {
	input string
	want  string
}

var sampleRx = regexp.MustCompile(`(?sm)^\s*want=([^\n]*)(?:\s+(.+\n))?\s*`)

func parseSample(funcName, comment string) (sample, bool) {
	text := strings.TrimPrefix(comment, "//")
	if v, ok := strings.CutPrefix(text, "/*"); ok {
		text = strings.TrimSuffix(v, "*/")
	}
	if m := sampleRx.FindStringSubmatch(text); m != nil {
		s := sample{
			want:  m[1],
			input: m[2],
		}
		return s, true
	}
	var zero sample
	return zero, false
}

func extractSamples(src []byte) map[string]sample {
	fs := token.NewFileSet()
	f, err := parser.ParseFile(fs, "aoc.go", src, parser.ParseComments)
	if err != nil {
		log.Fatalf("parsing source to extract samples: %v", err)
	}
	var lastInput string
	samples := make(map[string]sample)
	for _, d := range f.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Doc == nil {
			continue
		}
		funcName := fd.Name.Name
		for _, c := range fd.Doc.List {
			s, ok := parseSample(funcName, c.Text)
			if ok {
				s.input = Or(s.input, lastInput)
				samples[funcName] = s
				lastInput = s.input
				break
			}
		}
	}
	return samples
}

type Puzzle struct {
	year       int
	day        day
	SampleMode bool

	solver  partSolver
	samples map[string]sample
}

func (p *Puzzle) Description() []byte {
	return fileOrFetch(fmt.Sprintf("%d/%d.html", p.year, p.day.day), fmt.Sprintf("https://adventofcode.com/%d/day/%d", p.year, p.day.day))
}

func (p *Puzzle) Input() []byte {
	if p.SampleMode {
		return []byte(p.Sample().input)
	}
	return fileOrFetch(fmt.Sprintf("%d/%d.input", p.year, p.day.day), fmt.Sprintf("https://adventofcode.com/%d/day/%d/input", p.year, p.day.day))
}

func (p *Puzzle) Scanner() *bufio.Scanner {
	return bufio.NewScanner(bytes.NewReader(p.Input()))
}

func (p *Puzzle) ForLinesY(onLine func(int, string)) {
	s := p.Scanner()
	y := -1
	for s.Scan() {
		y++
		onLine(y, s.Text())
	}
	if err := s.Err(); err != nil {
		log.Fatal(err)
	}
}

func (p *Puzzle) Debug(v ...any) {
	if flagDebug {
		fmt.Println(v...)
	}
}

func (p *Puzzle) Debugf(format string, args ...any) {
	if flagDebug && p.SampleMode {
		fmt.Printf(format+"\n", args...)
	}
}

// ForLines calls onLine for each line of input.
// The y value is the row number, starting with 0.
func (p *Puzzle) ForLines(onLine func(line string)) {
	p.ForLinesY(func(_ int, line string) { onLine(line) })
}

func (p *Puzzle) Sample() sample {
	sample, ok := p.samples[p.solver.Name]
	if !ok {
		log.Fatalf("no sample found for %v", p.solver.Name)
	}
	return sample
}

type day struct {
	day   int
	parts []partSolver
}

type partSolver struct {
	fn   func() any
	Part string
	Name string
}

// extractMethods registers a struct with methods named D{day}p{part} for
// each day/part of Advent of Code. The methods must match the
// signature of PuzzleSolver.
func extractMethods(x any) map[int]day {
	rx := regexp.MustCompile(`^D(\d+)p(\d+.*)$`)
	v := reflect.ValueOf(x).Elem()
	if v.Kind() != reflect.Struct {
		log.Fatalf("Register: got %T; want struct", x)
	}
	vt := v.Type()
	byDays := map[int][]partSolver{}
	for i := 0; i < vt.NumMethod(); i++ {
		mt := vt.Method(i)
		mn := mt.Name
		matches := rx.FindStringSubmatch(mn)
		if len(matches) != 3 {
			continue
		}
		m := v.Method(i).Interface().(func() interface{})
		day, part := matches[1], matches[2]
		d := Int(day)
		byDays[d] = append(byDays[d], partSolver{
			fn:   m,
			Part: part,
			Name: mn,
		})
	}
	days := make(map[int]day, len(byDays))
	for d, parts := range byDays {
		slices.SortFunc(parts, func(i, j partSolver) int {
			return strings.Compare(i.Part, j.Part)
		})
		days[d] = day{parts: parts, day: d}
	}
	return days
}

var (
	flagCurDay     int
	flagPart       string
	flagDebug      bool
	flagOnlySample bool
	flagSkipSample bool
)

func init() {
	flag.IntVar(&flagCurDay, "day", -1, "day to run")
	flag.BoolVar(&flagOnlySample, "sample", false, "only run sample")
	flag.BoolVar(&flagSkipSample, "skip-sample", false, "skip sample")
	flag.BoolVar(&flagDebug, "debug", false, "debug mode")
	flag.StringVar(&flagPart, "part", "", "part to run")
}

var initFlags = sync.OnceFunc(flag.Parse)

func runDay(slvr any, year int, day day, samples map[string]sample) {
	p := Puzzle{
		year:    year,
		day:     day,
		samples: samples,
	}
	fmt.Println("Running day", day.day)
	sr := reflect.ValueOf(slvr)
	sr.Elem().FieldByName("Puzzle").Set(reflect.ValueOf(&p))
	for _, ps := range day.parts {
		p.solver = ps
		if flagPart != "" && ps.Part != flagPart {
			continue
		}

		for _, sm := range []bool{true, false} {
			if !sm && flagOnlySample {
				continue
			} else if sm && flagSkipSample {
				continue
			}
			p.SampleMode = sm
			if !sm {
				// Prime the input.
				p.Input()
			}
			t0 := time.Now()
			got := ps.fn()
			if sm {
				sample := p.Sample()
				if fmt.Sprint(got) != sample.want {
					fmt.Printf("part %s: %v ❌; want %v\n", ps.Part, got, sample.want)
					return
				}
				fmt.Printf("part %s sample: %v ✅ (%v) \n", ps.Part, got, time.Since(t0).Round(time.Microsecond))
			} else {
				fmt.Printf("part %s: %v (took %v) \n", ps.Part, got, time.Since(t0).Round(time.Microsecond))
			}
		}
	}
}

func Run(year int, src []byte, slvr any) {
	samples := extractSamples(src)
	days := extractMethods(slvr)
	initFlags()

	if flagCurDay != -1 {
		day, ok := days[flagCurDay]
		if !ok {
			log.Fatalf("no day %d", flagCurDay)
		}
		runDay(slvr, year, day, samples)
		return
	}

	dayNums := maps.Keys(days)
	slices.Sort(dayNums)
	for _, day := range dayNums {
		runDay(slvr, year, days[day], samples)
		fmt.Println()
	}
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

type Pt3[T constraints.Signed] struct {
	X, Y, Z T
}

type PtInt = Pt2[int]
type Pt3Int = Pt3[int]

func AbsDiff[T constraints.Signed](x, y T) T {
	v := x - y
	if v < 0 {
		v = -v
	}
	return v
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

var session = sync.OnceValue[string](func() string {
	return strings.TrimSpace(string(MustGet(os.ReadFile(filepath.Join(os.Getenv("HOME"), "keys", "aoc.session")))))
})

func request(method, url string, body io.Reader) *http.Request {
	req := MustGet(http.NewRequest(method, url, body))
	req.AddCookie(&http.Cookie{Name: "session", Value: session()})
	return req
}

func doRequest(req *http.Request) *http.Response {
	res := MustGet(http.DefaultClient.Do(req))
	if res.StatusCode != 200 {
		log.Fatalf("bad status fetching %s: %v", req.URL, res.Status)
	}
	return res
}

func fileOrFetch(filename, url string) []byte {
	if f, err := os.ReadFile(filename); err == nil {
		return f
	}

	body := fetch(url)
	MustDo(os.MkdirAll(filepath.Dir(filename), 0700))
	MustDo(os.WriteFile(filename, body, 0644))
	return body
}

func fetch(url string) []byte {
	res := doRequest(request("GET", url, nil))
	defer res.Body.Close()
	if res.StatusCode != 200 {
		panic(fmt.Sprintf("url %v failed: %v", url, res.Status))
	}
	return MustGet(io.ReadAll(res.Body))
}

// MustDo panics if err is non-nil.
func MustDo(err error) {
	if err != nil {
		panic(err)
	}
}

// MustGet returns v as is. It panics if err is non-nil.
func MustGet[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

func Int(s string) int {
	return MustGet(strconv.Atoi(strings.TrimSpace(s)))
}

func Ints(s ...string) []int {
	var out []int
	for _, v := range s {
		out = append(out, Int(v))
	}
	return out
}

func TrimPrefix(s, prefix string) string {
	s1, ok := strings.CutPrefix(s, prefix)
	if !ok {
		log.Fatalf("bad prefix: %q", s)
	}
	return s1
}

func Or[T any](list ...T) T {
	for _, v := range list {
		if !reflect.ValueOf(v).IsZero() {
			return v
		}
	}
	var zero T
	return zero
}

func Digit(r rune) int {
	if r < '0' || r > '9' {
		log.Fatalf("not a digit: %q", r)
	}
	return int(r - '0')
}

func Parallel[I, O any](in []I, f func(I) O) []O {
	var wg sync.WaitGroup
	wg.Add(len(in))
	out := make([]O, len(in))
	for i, v := range in {
		go func(i int, v I) {
			defer wg.Done()
			out[i] = f(v)
		}(i, v)
	}
	wg.Wait()
	return out
}

func Fold[T any, R any](in []T, f func(R, T) R, defVal R) R {
	out := defVal
	for _, v := range in {
		out = f(out, v)
	}
	return out
}

func ParallelMapFold[A, B, C any](in []A, f func(A) B, f2 func(C, B) C, defVal C) C {
	return Fold(
		Parallel(in, f),
		f2,
		defVal,
	)
}

func SolveQuad[T Number](a, b, c T) (float64, float64) {
	d := float64(b*b - 4*a*c)
	if d < 0 {
		log.Fatalf("no real roots")
	}
	d = math.Sqrt(d)
	a2 := float64(2 * a)
	return (-float64(b) + d) / a2, (-float64(b) - d) / a2
}

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

func GCD(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

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

func PolygonInnerArea(pts []Pt) int64 {
	var area int64

	for i := 1; i < len(pts); i++ {
		a := pts[i-1]
		b := pts[i]
		area += int64(a.X*b.Y) - int64(a.Y*b.X)
	}
	if area < 0 {
		area = -area
	}
	return area >> 1
}

func NewQueue[T any](in ...T) Queue[T] {
	return Queue[T]{
		q: in,
	}
}

type Queue[T any] struct {
	q []T
}

func (q *Queue[T]) Len() int {
	return len(q.q)
}

func (q *Queue[T]) Push(v T) {
	q.q = append(q.q, v)
}

func (q *Queue[T]) Pop() (T, bool) {
	if len(q.q) == 0 {
		var zero T
		return zero, false
	}
	v := q.q[0]
	q.q = q.q[1:]
	return v, true
}

func (q *Queue[T]) While(f func(T) bool) {
	for {
		v, ok := q.Pop()
		if !ok {
			return
		}
		if !f(v) {
			return
		}
	}
}

type Number interface {
	constraints.Float | constraints.Integer
}

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

func ParseBinary(in string) int64 {
	return MustGet(strconv.ParseInt(strings.TrimPrefix(in, "0b"), 2, 64))
}

type Segment struct {
	A, B Pt
}

func InitMap[K comparable, V any](m *map[K]V) {
	if *m == nil {
		*m = make(map[K]V)
	}
}

type Graph[K comparable] struct {
	Nodes map[K]bool
	Edges map[K]map[K]int
}

func (g *Graph[K]) Clone() *Graph[K] {
	var out Graph[K]
	out.Nodes = maps.Clone(g.Nodes)
	out.Edges = maps.Clone(g.Edges)
	for k, e := range g.Edges {
		out.Edges[k] = maps.Clone(e)
	}
	return &out
}

func (g *Graph[K]) NumPathsWithRestriction(start, end K, canVisit func(x K, alreadyVisited map[K]int) bool) int {
	return g.numPathsHelper(start, end, canVisit, make(map[K]int))
}

func (g *Graph[K]) NumPaths(start, end K) int {
	return g.NumPathsWithRestriction(start, end, func(x K, alreadyVisited map[K]int) bool {
		return alreadyVisited[x] == 0
	})
}

func (g *Graph[K]) numPathsHelper(start, end K, canVisit func(x K, alreadyVisited map[K]int) bool, visited map[K]int) int {
	if start == end {
		return 1
	}
	visited[start]++
	defer func() {
		visited[start]--
	}()
	count := 0
	for k := range g.Edges[start] {
		if canVisit(k, visited) {
			count += g.numPathsHelper(k, end, canVisit, visited)
		}
	}
	return count
}

func (g *Graph[K]) RemoveEdge(a, b K) {
	delete(g.Edges[a], b)
	delete(g.Edges[b], a)
}

func (g *Graph[K]) AllShortestPaths() map[Edge[K]]int {
	type key = Edge[K]
	dist := map[key]int{}
	for k1 := range g.Nodes {
		for k2 := range g.Nodes {
			if k1 == k2 {
				dist[key{k1, k1}] = 0
			} else if v, ok := g.Edges[k1][k2]; ok {
				dist[key{k1, k2}] = v
				dist[key{k2, k1}] = v
			} else {
				dist[key{k1, k2}] = math.MaxInt
				dist[key{k2, k1}] = math.MaxInt
			}
		}
	}
	for k1 := range g.Nodes {
		for k2 := range g.Nodes {
			for k3 := range g.Nodes {
				e12 := dist[key{k1, k2}]
				e23 := dist[key{k2, k3}]
				e13 := dist[key{k1, k3}]
				if e12 == math.MaxInt || e23 == math.MaxInt {
					continue
				}
				if e := e12 + e23; e < e13 {
					dist[key{k1, k3}] = e
				}
			}
		}
	}
	return dist
}

func (g *Graph[K]) ReachableNodes(a K) map[K]bool {
	visited := make(map[K]bool)
	var q Queue[K]
	q.Push(a)
	q.While(func(v K) bool {
		if visited[v] {
			return true
		}
		visited[v] = true
		for k := range g.Edges[v] {
			q.Push(k)
		}
		return true
	})
	return visited
}

func (g *Graph[K]) AddNode(a K) {
	if g.Nodes == nil {
		g.Nodes = make(map[K]bool)
	}
	g.Nodes[a] = true
}

func (g *Graph[K]) RemoveNode(a K) {
	for e := range g.Edges[a] {
		delete(g.Edges[e], a)
	}
	delete(g.Edges, a)
	delete(g.Nodes, a)
}

func (g *Graph[K]) AddEdge(a, b K, dist int) {
	InitMap(&g.Edges)
	InitMap(&g.Nodes)
	if g.Edges[a] == nil {
		g.Edges[a] = make(map[K]int)
	}
	if g.Edges[b] == nil {
		g.Edges[b] = make(map[K]int)
	}
	g.Edges[a][b] = dist
	g.Edges[b][a] = dist
	if g.Nodes == nil {
		g.Nodes = make(map[K]bool)
	}
	g.Nodes[a] = true
	g.Nodes[b] = true
}

func (g Graph[K]) LongestPath(start, end K) (rp int, ok bool) {
	return g.longestPathHelper(start, end, make(map[K]bool))
}

func (g Graph[K]) longestPathHelper(start, end K, visited map[K]bool) (rp int, ok bool) {
	if start == end {
		return 0, true
	}

	visited[start] = true
	defer func() {
		visited[start] = false
	}()
	max := -1
	for k, v := range g.Edges[start] {
		if visited[k] {
			continue
		}
		got, ok := g.longestPathHelper(k, end, visited)
		got += v
		if ok && (max == -1 || got > max) {
			max = got
		}
	}
	if max != -1 {
		return max, true
	}
	return 0, false
}

func (g *Graph[K]) Collapse() {
	var zeroK K
	for {
		trimmed := false
		for k1, e := range g.Edges {
			if len(e) == 2 {
				trimmed = true
				var k2, k3 K
				var d2, d3 int
				for k, v := range e {
					if k2 == zeroK {
						k2 = k
						d2 = v
					} else {
						k3 = k
						d3 = v
						break
					}
				}

				delete(g.Edges, k1)
				delete(g.Nodes, k1)
				g.RemoveEdge(k2, k1)
				g.RemoveEdge(k3, k1)
				g.AddEdge(k2, k3, d2+d3)
			}
		}
		if !trimmed {
			break
		}
	}
}

type Edge[T comparable] struct {
	A, B T
}

// AnyKey returns any key from the map.
// It panics if the map is empty.
func AnyKey[K comparable, V any](m map[K]V) K {
	for k := range m {
		return k
	}
	panic("bad")
}

// MinCut calculates the minimum cut of a graph using the Stoer–Wagner
// algorithm. It returns a list of edges that make up the cut.
func (g *Graph[T]) MinCut() []Edge[T] {
	var (
		g2 = g.Clone() // copy of graph to mutate

		start = AnyKey(g2.Nodes) // any node

		set = map[T][]T{}

		curMaxSet     T
		curMaxSetSize int

		minCut = math.MaxInt
		maxSet []T
	)
	for len(g2.Nodes) > 2 {
		s, t, w := g2.minCutPhase(start)
		if w < minCut {
			minCut = w
			maxSet = slices.Clone(set[curMaxSet])
		}
		if _, ok := set[s]; !ok {
			set[s] = []T{s}
		}
		if st, ok := set[t]; !ok {
			set[s] = append(set[s], t)
		} else {
			set[s] = append(set[s], st...)
			delete(set, t)
		}

		if len(set[s]) > curMaxSetSize {
			curMaxSet = s
			curMaxSetSize = len(set[s])
		}
		g2.merge(s, t)
	}

	var cuts []Edge[T]
	for _, v := range maxSet {
		for e := range g.Edges[v] {
			if !slices.Contains(maxSet, e) {
				cuts = append(cuts, Edge[T]{v, e})
			}
		}
	}
	if len(cuts) != minCut {
		panic(fmt.Sprintf("reconstructed cuts = %d; want %d", len(cuts), minCut))
	}
	return cuts
}

// minCutPhase runs one phase of the min cut algorithm. It returns the last two
// nodes traversed and the weight of the cut.
//
// It is equivalent to running a max flow algorithm from start to any other node
// in the graph.
func (g *Graph[T]) minCutPhase(start T) (s, t T, wOut int) {
	var pq PQ[T]
	var pris = map[T]*PQI[T]{}
	for k := range g.Nodes {
		i := &PQI[T]{
			V: k,
			P: 0,
		}
		if k == start {
			i.P = 1
		}
		pris[k] = i
		pq.Push(i)
	}

	for pq.Len() > 0 {
		next := pq.Pop()

		for k, v := range g.Edges[next.V] {
			p := pris[k]
			p.P += v
			if p.Index() != -1 {
				pq.Update(p)
			}
		}
		s, t = t, next.V
		wOut = next.P
	}
	return
}

func (g *Graph[T]) merge(s, t T) {
	for k, tvk := range g.Edges[t] {
		svk := g.Edges[s][k]
		g.RemoveEdge(t, k)
		g.AddEdge(s, k, svk+tvk)
	}
	g.RemoveEdge(s, s)
	delete(g.Nodes, t)
	delete(g.Edges, t)
}

type Stack[T any] struct {
	s []T
}

func (s *Stack[T]) Push(v T) {
	s.s = append(s.s, v)
}

func (s *Stack[T]) Pop() (T, bool) {
	if len(s.s) == 0 {
		var zero T
		return zero, false
	}
	v := s.s[len(s.s)-1]
	s.s = s.s[:len(s.s)-1]
	return v, true
}

func (s *Stack[T]) While(f func(T) bool) {
	for {
		v, ok := s.Pop()
		if !ok {
			return
		}
		if !f(v) {
			return
		}
	}
}

func (s *Stack[T]) Peek() (T, bool) {
	if len(s.s) == 0 {
		var zero T
		return zero, false
	}
	return s.s[len(s.s)-1], true
}

func Digits(line string) []int {
	var in []int
	for _, c := range line {
		in = append(in, Digit(c))
	}
	return in
}

type PQI[T any] struct {
	V  T
	P  int
	ix int
}

func (i *PQI[T]) String() string {
	return fmt.Sprintf("%v:%v", i.V, i.P)
}

func (i *PQI[T]) Index() int {
	return i.ix
}

type PQ[T any] struct {
	pq pq[T]
}

func (pq *PQ[T]) Push(v *PQI[T]) {
	heap.Push(&pq.pq, v)
}

func (pq *PQ[T]) Pop() *PQI[T] {
	return heap.Pop(&pq.pq).(*PQI[T])
}

func (pq *PQ[T]) Update(v *PQI[T]) {
	heap.Fix(&pq.pq, v.ix)
}

func (pq *PQ[T]) Peek() *PQI[T] {
	return pq.pq[0]
}

func (pq *PQ[T]) Len() int {
	return pq.pq.Len()
}

type pq[T any] []*PQI[T]

func (pq pq[T]) Len() int { return len(pq) }

func (pq pq[T]) Less(i, j int) bool {
	// We want Pop to give us the highest, not lowest, priority so we use greater than here.
	return pq[i].P > pq[j].P
}

func (pq pq[T]) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].ix = i
	pq[j].ix = j
}

func (pq *pq[T]) Push(x any) {
	n := len(*pq)
	i := x.(*PQI[T])
	i.ix = n
	*pq = append(*pq, i)
}

func (pq *pq[T]) Pop() any {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil // avoid memory leak
	item.ix = -1   // for safety

	*pq = old[0 : n-1]
	return item
}
