package aoc

import "testing"

func TestPolygonArea(t *testing.T) {
	tests := []struct {
		pts  []Pt
		want int64
	}{
		{
			pts: []Pt{
				{X: 0, Y: 0},
				{X: 5, Y: 0},
				{X: 5, Y: 5},
				{X: 0, Y: 5},
				{X: 0, Y: 0},
			},
			want: 25,
		},
	}

	for _, tt := range tests {
		if got := PolygonInnerArea(tt.pts); got != tt.want {
			t.Errorf("PolygonArea(%v) = %v, want %v", tt.pts, got, tt.want)
		}
	}
}

func TestParseSample(t *testing.T) {
	tests := []struct {
		comment string
		want    sample
	}{
		{
			comment: `/*
want=1

some-input
*/`,
			want: sample{
				want: "1",
				input: `some-input
`,
			},
		},

		{
			comment: `/*
want=1234

multi-line-input
other-line
other-line-2
*/`,
			want: sample{
				want: "1234",
				input: `multi-line-input
other-line
other-line-2
`,
			},
		},
		{
			comment: `/*
want=1234

multi-line-input
other-line
other-line-2
*/`,
			want: sample{
				want: "1234",
				input: `multi-line-input
other-line
other-line-2
`,
			},
		},
	}

	for _, tt := range tests {
		if got, ok := parseSample("foo", tt.comment); !ok || got != tt.want {
			t.Errorf("ParseSample = %v, want %v", got, tt.want)
		}
	}
}
