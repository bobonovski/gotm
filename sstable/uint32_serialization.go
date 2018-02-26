package sstable

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	log "github.com/golang/glog"
)

// serialize data to file
func Uint32Serialize(m *Uint32Matrix, fn string) error {
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
func Uint32Deserialize(fn string) (*Uint32Matrix, error) {
	file, err := os.Open(fn)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	lineIdx := 0
	var tmp *Uint32Matrix

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		txt := scanner.Text()
		if lineIdx == 0 {
			shape := strings.Split(txt, ",")
			if len(shape) != 2 {
				return nil, fmt.Errorf("model corrupted, shape not found: %s", txt)
			}
			row, err := strconv.ParseUint(shape[0], 10, 32)
			if err != nil {
				return nil, err
			}
			col, err := strconv.ParseUint(shape[1], 10, 32)
			if err != nil {
				return nil, err
			}
			tmp = NewUint32Matrix(uint32(row), uint32(col))
			lineIdx += 1
			continue
		}

		value := strings.Split(txt, ",")
		if len(value) != 3 {
			log.Infof("data corrupted, row %d, data %s",
				lineIdx, txt)
			continue
		}
		ridx, err := strconv.ParseUint(value[0], 10, 32)
		if err != nil {
			return nil, err
		}
		cidx, err := strconv.ParseUint(value[1], 10, 32)
		if err != nil {
			return nil, err
		}
		val, err := strconv.ParseUint(value[2], 10, 32)
		if err != nil {
			return nil, err
		}
		tmp.Set(uint32(ridx), uint32(cidx), uint32(val))

		lineIdx += 1
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return tmp, nil
}
