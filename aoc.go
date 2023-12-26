// Package aoc are quick & dirty utilities for helping Maisem
// solve Advent of Code problems. (forked from bradfitz/aoc)
package aoc

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"slices"
	"strings"
	"sync"
	"time"

	"golang.org/x/exp/maps"
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

// Or returns the first non-zero value in list.
func Or[T any](list ...T) T {
	for _, v := range list {
		if !reflect.ValueOf(v).IsZero() {
			return v
		}
	}
	var zero T
	return zero
}

// Parallel runs f in parallel for each value in in.
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

// Fold folds the input into a single value.
func Fold[T any, R any](in []T, f func(R, T) R, defVal R) R {
	out := defVal
	for _, v := range in {
		out = f(out, v)
	}
	return out
}

// ParallelMapFold maps the input to a new slice, then folds the result.
// It is equivalent to Fold(Parallel(in, f), f2, defVal).
func ParallelMapFold[A, B, C any](in []A, f func(A) B, f2 func(C, B) C, defVal C) C {
	return Fold(
		Parallel(in, f),
		f2,
		defVal,
	)
}

// InitMap initializes a map if it is nil.
func InitMap[K comparable, V any](m *map[K]V) {
	if *m == nil {
		*m = make(map[K]V)
	}
}

// AnyKey returns any key from the map.
// It panics if the map is empty.
func AnyKey[K comparable, V any](m map[K]V) K {
	for k := range m {
		return k
	}
	panic("bad")
}

// TrimPrefix trims the prefix from s. It panics if the prefix is not found.
func TrimPrefix(s, prefix string) string {
	out, ok := strings.CutPrefix(s, prefix)
	if !ok {
		panic(fmt.Sprintf("bad prefix: %q", prefix))
	}
	return out
}
