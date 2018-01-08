package sstable

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSortedMap(t *testing.T) {
	m := NewSortedMap(uint32(10))

	// test minimum representational bits
	assert.Equal(t, uint32(4), m.RotateLen)

	m.Update(uint32(123), uint32(1), uint32(4))
	assert.Equal(t, uint32(65), m.Data[uint32(123)][0])

	tid, count := m.Get(uint32(123), 0)
	assert.Equal(t, uint32(4), count)
	assert.Equal(t, uint32(1), tid)

	m.Update(uint32(123), uint32(2), uint32(6))
	assert.Equal(t, uint32(98), m.Data[uint32(123)][0])
	assert.Equal(t, uint32(65), m.Data[uint32(123)][1])

	m.Incr(uint32(123), uint32(1), uint32(1))
	assert.Equal(t, uint32(81), m.Data[uint32(123)][1])
}
