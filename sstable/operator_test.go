package sstable

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUint32VectorSum(t *testing.T) {
	v := []uint32{3, 4, 5}
	assert.Equal(t, uint32(12), Uint32VectorSum(v))
}

func TestFloat32VectorSum(t *testing.T) {
	v := []float32{1.0, 2.0, 3.0}
	assert.Equal(t, float32(6.0), Float32VectorSum(v))
}
