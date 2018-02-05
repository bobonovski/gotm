package sstable

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSortedMap(t *testing.T) {
	m := NewSortedMap(uint32(10))

	// test minimum representational bits
	assert.Equal(t, uint32(4), m.RotateLen)

	m.Incr(uint32(123), uint32(1), uint32(4))
	assert.Equal(t, uint32(65), m.Data[uint32(123)][0])

	tid, count := m.Get(uint32(123), 0)
	assert.Equal(t, uint32(1), tid)
	assert.Equal(t, uint32(4), count)

	m.Incr(uint32(123), uint32(2), uint32(6))
	tid, count = m.Get(uint32(123), 0)
	assert.Equal(t, uint32(2), tid)
	assert.Equal(t, uint32(6), count)

	m.Incr(uint32(123), uint32(1), uint32(4))
	tid, count = m.Get(uint32(123), 0)
	assert.Equal(t, uint32(1), tid)
	assert.Equal(t, uint32(8), count)

	m.Decr(uint32(123), uint32(1), uint32(1))
	tid, count = m.Get(uint32(123), 0)
	assert.Equal(t, uint32(1), tid)
	assert.Equal(t, uint32(7), count)
}
