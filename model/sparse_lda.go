package model

import (
	"math"
	"math/rand"

	log "github.com/golang/glog"

	"github.com/bobonovski/gotm/corpus"
	"github.com/bobonovski/gotm/sstable"
)

func init() {
	Register("sparselda", NewSparseLDA)
}

type SparseLDA struct {
	*LDA
	wtm *sstable.SortedMap
}

// NewSparseLDA creates a sparse lda instance with time
// and memory efficient gibbs sampler
func NewSparseLDA(dat *corpus.Corpus,
	topicNum uint32, alpha float32, beta float32) Model {
	return &SparseLDA{
		LDA: NewLDA(dat, topicNum, alpha, beta).(*LDA),
		wtm: sstable.NewSortedMap(topicNum),
	}
}

func (this *SparseLDA) Train(iter int) {
	this.Init()
	row, col := this.wt.Shape()
	for r := uint32(0); r < row; r += 1 {
		for c := uint32(0); c < col; c += 1 {
			cnt := this.wt.Get(r, c)
			if cnt > 0 {
				this.wtm.Incr(r, c, cnt)
			}
		}
	}
	this.wt = nil

	dw := sstable.DocWord{}

	// compute smoothing bucket
	smoothingBucket := float32(0.0)
	for k := uint32(0); k < this.TopicNum; k += 1 {
		smoothingBucket += (this.Alpha * this.Beta) /
			(this.Beta*float32(this.Data.VocabSize) +
				float32(this.wts.Get(k, uint32(0))))
	}

	// word-topic bucket cache
	wtbCache := make([]float32, this.TopicNum)
	for iterIdx := 0; iterIdx < iter; iterIdx += 1 {
		if iterIdx%10 == 0 {
			log.Infof("iter %5d, likelihood %f", iterIdx, this.Likelihood())
		}

		// fast sparse gibbs sampling
		for doc, wcs := range this.Data.Docs {
			// document-topic bucket
			docTopicBucket := float32(0.0)

			for k := uint32(0); k < this.TopicNum; k += 1 {
				docTopicBucket += (this.Beta * float32(this.dt.Get(doc, k))) /
					(this.Beta*float32(this.Data.VocabSize) +
						float32(this.wts.Get(k, uint32(0))))
				wtbCache[k] = (this.Alpha + float32(this.dt.Get(doc, k))) /
					(this.Beta*float32(this.Data.VocabSize) +
						float32(this.wts.Get(k, uint32(0))))
			}

			for i, w := range corpus.ExpandWords(wcs) {
				// get the current topic of word w
				dw.DocId = doc
				dw.WordIdx = uint32(i)
				k := this.dwt[dw]

				// subtract old value from buckets
				denom := (this.Beta*float32(this.Data.VocabSize) +
					float32(this.wts.Get(k, uint32(0))))
				smoothingBucket -= (this.Alpha * this.Beta) / denom
				docTopicBucket -= (this.Beta * float32(this.dt.Get(doc, k))) / denom

				// decrease corresponding sufficient statistics
				this.wtm.Decr(w, k, uint32(1))
				this.dt.Decr(doc, k, uint32(1))
				this.wts.Decr(k, uint32(0), uint32(1))

				// update bucket values
				denom = (this.Beta*float32(this.Data.VocabSize) +
					float32(this.wts.Get(k, uint32(0))))
				smoothingBucket += (this.Alpha * this.Beta) / denom
				docTopicBucket += (this.Beta * float32(this.dt.Get(doc, k))) / denom
				wtbCache[k] = (this.Alpha + float32(this.dt.Get(doc, k))) / denom

				// compute word-topic bucket sum
				wtbSum := float32(0.0)
				for idx, _ := range this.wtm.Data[w] {
					tid, count := this.wtm.Get(w, idx)
					wtbSum += wtbCache[tid] * float32(count)
				}
				dtbSum := docTopicBucket
				sbSum := smoothingBucket

				// resample topic assignment
				var cumsum float32
				u := rand.Float32() * (wtbSum + dtbSum + sbSum)
				if u < wtbSum { // topic-word bucket
					cumsum = 0.0
					for tcIdx, _ := range this.wtm.Data[w] {
						tid, count := this.wtm.Get(w, tcIdx)
						cumsum += wtbCache[tid] * float32(count)
						if cumsum >= u {
							k = tid
							break
						}
					}
				} else if u < (wtbSum+dtbSum) && u >= wtbSum { // doc-topic bucket
					cumsum = 0.0
					u = u - wtbSum
					for kidx := uint32(0); kidx < this.TopicNum; kidx += 1 {
						cumsum += (this.Beta * float32(this.dt.Get(doc, k))) / denom
						if cumsum >= u {
							k = kidx
							break
						}
					}
				} else { // smoothing bucket
					cumsum = 0.0
					for kidx := uint32(0); kidx < this.TopicNum; kidx += 1 {
						cumsum += (this.Alpha * this.Beta) / denom
						if cumsum >= u {
							k = kidx
							break
						}
					}
				}

				denom = (this.Beta*float32(this.Data.VocabSize) +
					float32(this.wts.Get(k, uint32(0))))
				smoothingBucket -= (this.Alpha * this.Beta) / denom
				docTopicBucket -= (this.Beta * float32(this.dt.Get(doc, k))) / denom

				// increase corresponding sufficient statistics
				this.wtm.Incr(w, k, uint32(1))
				this.dt.Incr(doc, k, uint32(1))
				this.wts.Incr(k, uint32(0), uint32(1))
				this.dwt[dw] = k

				// update bucket values
				denom = (this.Beta*float32(this.Data.VocabSize) +
					float32(this.wts.Get(k, uint32(0))))
				smoothingBucket += (this.Alpha * this.Beta) / denom
				docTopicBucket += (this.Beta * float32(this.dt.Get(doc, k))) / denom
				wtbCache[k] = (this.Alpha + float32(this.dt.Get(doc, k))) / denom
			}
		}
	}
}

// infer topics on new documents
func (this *SparseLDA) Infer(iter int) {
	this.Train(iter)
}

// compute the posterior point estimation of word-topic mixture
// beta (Dirichlet prior) + data -> phi
func (this *SparseLDA) Phi() *sstable.Float32Matrix {
	phi := sstable.NewFloat32Matrix(this.Data.VocabSize, this.TopicNum)

	for w := uint32(0); w < this.Data.VocabSize; w += 1 {
		// convert sparse vector to dense vector
		wordTopicCount := make([]uint32, this.TopicNum)
		for tcIdx, _ := range this.wtm.Data[w] {
			topicId, count := this.wtm.Get(w, tcIdx)
			wordTopicCount[topicId] = count
		}
		for k := uint32(0); k < this.TopicNum; k += 1 {
			result := (float32(wordTopicCount[k]) + this.Beta) /
				(float32(this.wts.Get(k, uint32(0))) +
					float32(this.Data.VocabSize)*this.Beta)
			phi.Set(w, k, result)
		}
	}

	return phi
}

// serialize word-topic distribution
func (this *SparseLDA) SavePhi(fn string) error {
	phi := this.Phi()
	if err := phi.Serialize(fn + ".phi"); err != nil {
		return err
	}
	return nil
}

// compute the joint likelihood of corpus
func (this *SparseLDA) Likelihood() float64 {
	phi := this.Phi()
	theta := this.Theta()

	sum := float64(0.0)
	for doc, wcs := range this.Data.Docs {
		for _, w := range corpus.ExpandWords(wcs) {
			topicSum := float32(0.0)
			for k := uint32(0); k < this.TopicNum; k += 1 {
				topicSum += phi.Get(w, k) * theta.Get(doc, k)
			}
			sum += math.Log(float64(topicSum))
		}
	}

	return sum
}

// serialize word-topic matrix
func (this *SparseLDA) SaveWordTopic(fn string) error {
	if err := this.wtm.Serialize(fn); err != nil {
		return err
	}
	return nil
}

// deserialize word-topic matrix
func (this *SparseLDA) LoadWordTopic(fn string) error {
	if err := this.wtm.Deserialize(fn); err != nil {
		return err
	}
	return nil
}
