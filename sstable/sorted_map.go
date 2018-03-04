package sstable

import (
	"bufio"
	"errors"
	"fmt"
	"log"
	"math/bits"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
)

type SortedMap struct {
	Data       map[uint32][]uint32
	RotateLen  uint32
	TopicMask  uint32
	MaxWordId  uint32
	MaxTopicId uint32
}

func NewSortedMap(topicNum uint32) *SortedMap {
	rotateLen := uint32(bits.Len32(topicNum))
	return &SortedMap{
		Data:      make(map[uint32][]uint32),
		RotateLen: rotateLen,
		TopicMask: (uint32(1) << rotateLen) - 1,
	}
}

var (
	// Cache nonzero topic count of word and the values
	// are sorted by count in descending order. Each value
	// is represented by a uint32 number where the lower k
	// bits are the minimum bits needed to hold the max number
	// of topics and the upper 32-k bits are used to hold the
	// count number
	WordTopicMap *SortedMap
)

// serialize data to file
func (this *SortedMap) Serialize(fn string) error {
	out, err := os.OpenFile(fn, os.O_WRONLY|os.O_TRUNC|os.O_CREATE, os.ModePerm)
	if err != nil {
		return err
	}
	defer out.Close()

	if len(this.Data) == 0 {
		return nil
	}

	// write the matrix shape
	out.WriteString(fmt.Sprintf("%d,%d\n",
		this.MaxWordId+uint32(1), this.MaxTopicId+uint32(1)))

	for w := uint32(0); w <= this.MaxWordId; w += 1 {
		for i, _ := range this.Data[w] {
			topicId, count := this.Get(w, i)
			out.WriteString(fmt.Sprintf("%d,%d,%d\n", w, topicId, count))
		}
	}
	return nil
}

// deserialize data from file
func (this *SortedMap) Deserialize(fn string) error {
	file, err := os.Open(fn)
	if err != nil {
		return err
	}
	defer file.Close()

	var lineIdx int
	var tmp *SortedMap

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
			tmp = NewSortedMap(uint32(row))
			tmp.MaxWordId = uint32(row) - uint32(1)
			tmp.MaxTopicId = uint32(col) - uint32(1)
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

		tmp.Incr(uint32(ridx), uint32(cidx), uint32(val))

		lineIdx += 1
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	this = tmp

	return nil
}

// get the i-th element of the value slice of wordId and return
// parsed value of topicId and count
func (this *SortedMap) Get(wordId uint32, idx int) (uint32, uint32) {
	if idx > len(this.Data[wordId]) {
		panic(ErrIndexOutOfRange)
	}
	val := atomic.LoadUint32(&this.Data[wordId][idx])
	count := val >> this.RotateLen
	topicId := val & this.TopicMask
	return topicId, count
}

func (this *SortedMap) Incr(wordId uint32, topicId uint32, count uint32) {
	if count == 0 {
		return
	}

	idx := -1
	for i, v := range this.Data[wordId] {
		if v&this.TopicMask == topicId {
			idx = i
			break
		}
	}
	if idx == -1 {
		if wordId > this.MaxWordId {
			this.MaxWordId = wordId
		}
		if topicId > this.MaxTopicId {
			this.MaxTopicId = topicId
		}
		this.Data[wordId] = append(this.Data[wordId],
			(count<<this.RotateLen)+topicId)
		for k := len(this.Data[wordId]) - 1; k > 0; k -= 1 {
			if this.Data[wordId][k] > this.Data[wordId][k-1] {
				this.Data[wordId][k], this.Data[wordId][k-1] =
					this.Data[wordId][k-1], this.Data[wordId][k]
				continue
			}
			break
		}
	} else {
		// update count
		_, oldCount := this.Get(wordId, idx)
		this.Data[wordId][idx] = ((count + oldCount) << this.RotateLen) + topicId

		// sort the values using bubble sort
		for k := idx; k > 0; k -= 1 {
			if this.Data[wordId][k] > this.Data[wordId][k-1] {
				this.Data[wordId][k], this.Data[wordId][k-1] =
					this.Data[wordId][k-1], this.Data[wordId][k]
				continue
			}
			break
		}
	}
}

func (this *SortedMap) Decr(wordId uint32, topicId uint32, count uint32) {
	if _, ok := this.Data[wordId]; !ok {
		return
	}
	if count == 0 {
		return
	}

	idx := -1
	for i, _ := range this.Data[wordId] {
		if this.Data[wordId][i]&this.TopicMask == topicId {
			idx = i
			break
		}
	}
	if idx == -1 {
		return
	}

	// update count
	_, oldCount := this.Get(wordId, idx)
	if count > oldCount {
		count = oldCount
	}
	if oldCount-count == 0 { // delete the topic count
		curLen := len(this.Data[wordId])
		// move all the smaller value forward
		for k := idx + 1; k < len(this.Data[wordId]); k += 1 {
			this.Data[wordId][k-1] = this.Data[wordId][k]
		}
		// shrink the slice
		this.Data[wordId] = this.Data[wordId][0 : curLen-1]
	} else {
		this.Data[wordId][idx] = ((oldCount - count) << this.RotateLen) + topicId
		// sort the values using bubble sort
		for k := idx; k < len(this.Data[wordId])-1; k += 1 {
			if this.Data[wordId][k] < this.Data[wordId][k+1] {
				this.Data[wordId][k], this.Data[wordId][k+1] =
					this.Data[wordId][k+1], this.Data[wordId][k]
				continue
			}
			break
		}
	}

}
