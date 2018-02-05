package sstable

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFloat32MatrixShape(t *testing.T) {
	m := NewFloat32Matrix(uint32(2), uint32(3))

	r, c := m.Shape()

	assert.Equal(t, uint32(2), r)
	assert.Equal(t, uint32(3), c)
}

func TestFloat32MatrixGet(t *testing.T) {
	m := NewFloat32Matrix(uint32(2), uint32(3))

	val := float32(0.0)
	for r := 0; r < 2; r += 1 {
		for c := 0; c < 3; c += 1 {
			m.Set(uint32(r), uint32(c), val)
			val += float32(1.0)
		}
	}

	assert.Equal(t, float32(0), m.Get(0, 0))
	assert.Equal(t, float32(1), m.Get(0, 1))
	assert.Equal(t, float32(2), m.Get(0, 2))
	assert.Equal(t, float32(3), m.Get(1, 0))
	assert.Equal(t, float32(4), m.Get(1, 1))
	assert.Equal(t, float32(5), m.Get(1, 2))
}
