package aoc

import (
	"container/heap"
	"fmt"
)

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

func MinQueue[T any]() *PQ[T] {
	return &PQ[T]{
		pq: pq[T]{
			min: true,
		},
	}
}

func MaxQueue[T any]() *PQ[T] {
	return &PQ[T]{
		pq: pq[T]{
			min: true,
		},
	}
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
	return pq.pq.q[0]
}

func (pq *PQ[T]) Len() int {
	return pq.pq.Len()
}

type pq[T any] struct {
	q   []*PQI[T]
	min bool
}

func (pq pq[T]) Len() int { return len(pq.q) }

func (pq pq[T]) Less(i, j int) bool {
	// We want Pop to give us the highest, not lowest, priority so we use greater than here.
	if pq.min {
		return pq.q[i].P < pq.q[j].P
	}
	return pq.q[i].P > pq.q[j].P
}

func (pq pq[T]) Swap(i, j int) {
	q := pq.q
	q[i], q[j] = q[j], q[i]
	q[i].ix = i
	q[j].ix = j
}

func (pq *pq[T]) Push(x any) {
	n := len(pq.q)
	i := x.(*PQI[T])
	i.ix = n
	pq.q = append(pq.q, i)
}

func (pq *pq[T]) Pop() any {
	old := pq.q
	n := len(old)
	item := old[n-1]
	old[n-1] = nil // avoid memory leak
	item.ix = -1   // for safety

	pq.q = old[0 : n-1]
	return item
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
