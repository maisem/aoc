package aoc

import (
	"fmt"
	"math"
	"slices"

	"golang.org/x/exp/maps"
)

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

// LongestPath returns the size of the longest path from start to end.
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

// Collapse collapses the graph by removing any nodes with only two edges and
// merging the two edges into one.
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
