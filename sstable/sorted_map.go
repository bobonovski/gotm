package sstable

import (
	"math/bits"

	"github.com/bobonovski/gotm/matrix"
)

type SortedMap struct {
	Data      map[uint32][]uint32
	RotateLen uint32
	TopicMask uint32
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

// get the i-th element of the value slice of wordId and return
// parsed value of topicId and count
func (this *SortedMap) Get(wordId uint32, idx int) (uint32, uint32) {
	if idx > len(this.Data[wordId]) {
		panic(matrix.ErrIndexOutOfRange)
	}
	count := this.Data[wordId][idx] >> this.RotateLen
	topicId := this.Data[wordId][idx] & this.TopicMask
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

// insert the tuple (wordId => (topicId, count)) to the map if not exists or
// update the tuple, then the value slice should be sorted
func (this *SortedMap) Update(wordId uint32, topicId uint32, count uint32) {
	if _, ok := this.Data[wordId]; !ok {
		this.Data[wordId] = append(this.Data[wordId],
			(count<<this.RotateLen)+topicId)
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
		if count == 0 {
			return
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
		if count == 0 { // delete the value
			curLen := len(this.Data[wordId])
			// move all the smaller value forward
			for k := idx + 1; k < len(this.Data[wordId]); k += 1 {
				this.Data[wordId][k-1] = this.Data[wordId][k]
			}
			// shrink the slice
			this.Data[wordId] = this.Data[wordId][0 : curLen-1]
		} else { // update value count
			// get the old value
			oldValue := this.Data[wordId][idx]
			this.Data[wordId][idx] = (count << this.RotateLen) + topicId

			// sort the values using bubble sort
			if this.Data[wordId][idx] < oldValue {
				for k := idx; k < len(this.Data[wordId])-1; k += 1 {
					if this.Data[wordId][k] < this.Data[wordId][k+1] {
						this.Data[wordId][k], this.Data[wordId][k+1] =
							this.Data[wordId][k+1], this.Data[wordId][k]
						continue
					}
					break
				}
			} else if this.Data[wordId][idx] > oldValue {
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
	}
}
