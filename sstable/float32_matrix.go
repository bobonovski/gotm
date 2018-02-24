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

// internal Float32 matrix representation
type Float32Matrix struct {
	nrow uint32
	ncol uint32
	data []float32
}

// NewFloat32Matrix creates a new Float32Matrix with r rows and c columns
// which is mainly used for caching temporary results
func NewFloat32Matrix(r, c uint32) *Float32Matrix {
	if r*c <= 0 {
		panic(ErrIndexOutOfRange)
	}
	return &Float32Matrix{
		nrow: r,
		ncol: c,
		data: make([]float32, r*c),
	}
}

// get the shape of the matrix
func (m *Float32Matrix) Shape() (uint32, uint32) {
	return m.nrow, m.ncol
}

// get the [r, c]-th element of the matrix
func (m *Float32Matrix) Get(r, c uint32) float32 {
	if r >= m.nrow || c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}
	return m.data[r*m.ncol+c]
}

// set val to the [r, c]-th element of the matrix
func (m *Float32Matrix) Set(r, c uint32, val float32) {
	if r >= m.nrow || c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}
	m.data[r*m.ncol+c] = val
}

// serialize data to file
func (m *Float32Matrix) Serialize(fn string) error {
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

	var val float32
	for ridx := uint32(0); ridx < r; ridx += 1 {
		for cidx := uint32(0); cidx < c; cidx += 1 {
			val = m.Get(ridx, cidx)
			if val > 0 { // only write out nonzero value
				out.WriteString(fmt.Sprintf("%d,%d,%e\n", ridx, cidx, val))
			}
		}
	}
	return nil
}

// deserialize data from file
func (m *Float32Matrix) Deserialize(fn string) error {
	file, err := os.Open(fn)
	if err != nil {
		return err
	}
	defer file.Close()

	var lineIdx int
	var tmp *Float32Matrix

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
			tmp = NewFloat32Matrix(uint32(row), uint32(col))
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
		val, err := strconv.ParseFloat(value[2], 32)
		if err != nil {
			return err
		}
		tmp.Set(uint32(ridx), uint32(cidx), float32(val))

		lineIdx += 1
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	tmp = m

	return nil
}
