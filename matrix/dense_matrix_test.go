package matrix

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDenseMatrixShape(t *testing.T) {
	m := NewDenseMatrix(uint32(2), uint32(3))

	r, c := m.Shape()

	assert.Equal(t, uint32(2), r)
	assert.Equal(t, uint32(3), c)
}

func TestDenseMatrixGet(t *testing.T) {
	m := NewDenseMatrix(uint32(2), uint32(3))

	val := uint32(0.0)
	for r := 0; r < 2; r += 1 {
		for c := 0; c < 3; c += 1 {
			m.Set(uint32(r), uint32(c), val)
			val += uint32(1.0)
		}
	}

	assert.Equal(t, uint32(0), m.Get(0, 0))
	assert.Equal(t, uint32(1), m.Get(0, 1))
	assert.Equal(t, uint32(2), m.Get(0, 2))
	assert.Equal(t, uint32(3), m.Get(1, 0))
	assert.Equal(t, uint32(4), m.Get(1, 1))
	assert.Equal(t, uint32(5), m.Get(1, 2))
}

func TestDenseMatrixIncrDecr(t *testing.T) {
	m := NewDenseMatrix(uint32(2), uint32(2))

	m.Incr(uint32(1), uint32(1), uint32(2))
	assert.Equal(t, uint32(2), m.Get(uint32(1), uint32(1)))

	m.Decr(uint32(1), uint32(1), uint32(1))
	assert.Equal(t, uint32(1), m.Get(uint32(1), uint32(1)))
}
