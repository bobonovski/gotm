package sstable

import (
	"bufio"
	"errors"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
)

var (
	ErrIndexOutOfRange = errors.New("matrix: index out of range")
	ErrBadShape        = errors.New("matrix: non-positive dimension not allowed")
)

// key-value pair util
type DocWord struct {
	DocId   uint32
	WordIdx uint32
}

// internal Uint32 matrix representation
type Uint32Matrix struct {
	nrow uint32
	ncol uint32
	data []uint32
}

// NewUint32Matrix creates a new Uint32Matrix with r rows and c columns.
// if r*c <= 0, it will panic. A uint32 slice is used as the underlying
// storage and the data layout is in row major order, i.e. the (i*c + j)-th
// element in the data slice is the [i, j]-th element in the matrix.
// Vector is defined as a matrix one column, i.e. a column vector.
func NewUint32Matrix(r, c uint32) *Uint32Matrix {
	if r*c <= 0 {
		panic(ErrIndexOutOfRange)
	}
	return &Uint32Matrix{
		nrow: r,
		ncol: c,
		data: make([]uint32, r*c),
	}
}

// get the shape of the matrix
func (m *Uint32Matrix) Shape() (uint32, uint32) {
	return m.nrow, m.ncol
}

// get the [r, c]-th element of the matrix
func (m *Uint32Matrix) Get(r, c uint32) uint32 {
	if r >= m.nrow || c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}
	return m.data[r*m.ncol+c]
}

// get the r-th row of the matrix
func (m *Uint32Matrix) GetRow(r uint32) []uint32 {
	if r >= m.nrow {
		panic(ErrIndexOutOfRange)
	}

	var row []uint32
	for c := uint32(0); c < m.ncol; c += 1 {
		row = append(row, m.Get(r, c))
	}
	return row
}

// get the c-th column of the matrix
func (m *Uint32Matrix) GetCol(c uint32) []uint32 {
	if c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}

	var column []uint32
	for r := uint32(0); r < m.nrow; r += 1 {
		column = append(column, m.Get(r, c))
	}
	return column
}

// set val to the [r, c]-th element of the matrix
func (m *Uint32Matrix) Set(r, c uint32, val uint32) {
	if r >= m.nrow || c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}
	m.data[r*m.ncol+c] = val
}

// increment the [r, c]-th element of the matrix by val
func (m *Uint32Matrix) Incr(r, c uint32, val uint32) {
	if r >= m.nrow || c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}
	m.data[r*m.ncol+c] += val
}

// decrement the [r, c]-th element of the matrix by val
func (m *Uint32Matrix) Decr(r, c uint32, val uint32) {
	if r >= m.nrow || c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}
	m.data[r*m.ncol+c] -= val
}

// serialize data to file
func (m *Uint32Matrix) Serialize(fn string) error {
	out, err := os.OpenFile(fn, os.O_WRONLY|os.O_TRUNC|os.O_CREATE, os.ModePerm)
	if err != nil {
		return err
	}
	defer out.Close()

	r, c := m.Shape()
	if r*c == 0 {
		return nil
	}
	// write the matrix shape
	out.WriteString(fmt.Sprintf("%d,%d\n", r, c))

	var val uint32
	for ridx := uint32(0); ridx < r; ridx += 1 {
		for cidx := uint32(0); cidx < c; cidx += 1 {
			val = m.Get(ridx, cidx)
			if val > 0 { // only write out nonzero value
				out.WriteString(fmt.Sprintf("%d,%d,%d\n", ridx, cidx, val))
			}
		}
	}
	return nil
}

// deserialize data from file
func (m *Uint32Matrix) Deserialize(fn string) error {
	file, err := os.Open(fn)
	if err != nil {
		return err
	}
	defer file.Close()

	var lineIdx int

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		txt := scanner.Text()
		if lineIdx == 0 {
			shape := strings.Split(txt, ",")
			if len(shape) != 0 {
				return errors.New("model corrupted, shape not found")
			}
			row, err := strconv.ParseUint(shape[0], 10, 32)
			if err != nil {
				return err
			}
			col, err := strconv.ParseUint(shape[1], 10, 32)
			if err != nil {
				return err
			}
			m = NewUint32Matrix(uint32(row), uint32(col))
			continue
		}

		value := strings.Split(txt, ",")
		if len(value) != 3 {
			log.Printf("data corrupted, row %d, data %s",
				lineIdx, txt)
			continue
		}
		ridx, err := strconv.ParseUint(value[0], 10, 32)
		if err != nil {
			return err
		}
		cidx, err := strconv.ParseUint(value[1], 10, 32)
		if err != nil {
			return err
		}
		val, err := strconv.ParseUint(value[2], 10, 32)
		if err != nil {
			return err
		}
		m.Set(uint32(ridx), uint32(cidx), uint32(val))

		lineIdx += 1
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	return nil
}
