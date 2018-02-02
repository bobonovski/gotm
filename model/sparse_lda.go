package model

import (
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/bobonovski/gotm/corpus"
	"github.com/bobonovski/gotm/matrix"
	"github.com/bobonovski/gotm/sstable"
	"github.com/bobonovski/gotm/util"
)

type sparseLda struct {
	data     *corpus.Corpus
	alpha    float32 // document topic mixture hyperparameter
	beta     float32 // topic word mixture hyperparameter
	topicNum uint32
}

// NewSparseLDA creates a sparse lda instance with time
// and memory efficient gibbs sampler
func NewSparseLDA(dat *corpus.Corpus,
	topicNum uint32, alpha float32, beta float32) *sparseLda {
	// init sufficient statistics sstable
	sstable.DocTopic = matrix.NewUint32Matrix(dat.DocNum, topicNum)
	sstable.WordTopicSum = matrix.NewUint32Matrix(topicNum, uint32(1))
	sstable.DocWordTopic = make(map[sstable.DocWord]uint32)
	sstable.WordTopic = matrix.NewUint32Matrix(dat.VocabSize, topicNum)
	sstable.WordTopicMap = sstable.NewSortedMap(topicNum)

	return &sparseLda{
		data:     dat,
		alpha:    alpha,
		beta:     beta,
		topicNum: topicNum,
	}
}

func (this *sparseLda) Train(iter int) {
	// randomly assign topic to word
	rand.Seed(time.Now().Unix())
	dw := sstable.DocWord{}
	for doc, wcs := range this.data.Docs {
		for i, w := range corpus.ExpandWords(wcs) {
			// sample word topic
			k := uint32(rand.Int31n(int32(this.topicNum)))

			// update sufficient statistics
			sstable.DocTopic.Incr(doc, k, uint32(1))
			sstable.WordTopicSum.Incr(k, uint32(0), uint32(1))
			sstable.WordTopic.Incr(w, k, uint32(1))
			// update doc word topic assignment
			dw.DocId = doc
			dw.WordIdx = uint32(i)
			sstable.DocWordTopic[dw] = k
		}
	}
	row, col := sstable.WordTopic.Shape()
	for r := uint32(0); r < row; r += 1 {
		for c := uint32(0); c < col; c += 1 {
			cnt := sstable.WordTopic.Get(r, c)
			if cnt > 0 {
				sstable.WordTopicMap.Incr(r, c, cnt)
			}
		}
	}
	sstable.WordTopic = nil

	// compute smoothing bucket
	smoothingBucket := float32(0.0)
	for k := uint32(0); k < this.topicNum; k += 1 {
		smoothingBucket += (this.alpha * this.beta) /
			(this.beta*float32(this.data.VocabSize) +
				float32(sstable.WordTopicSum.Get(k, uint32(0))))
	}

	// word-topic bucket cache
	wtbCache := make([]float32, this.topicNum)
	for iterIdx := 0; iterIdx < iter; iterIdx += 1 {
		if iterIdx%10 == 0 {
			log.Printf("iter %5d, likelihood %f", iterIdx, this.Likelihood())
		}

		// fast sparse gibbs sampling
		for doc, wcs := range this.data.Docs {
			// document-topic bucket
			docTopicBucket := float32(0.0)

			for k := uint32(0); k < this.topicNum; k += 1 {
				docTopicBucket += (this.beta * float32(sstable.DocTopic.Get(doc, k))) /
					(this.beta*float32(this.data.VocabSize) +
						float32(sstable.WordTopicSum.Get(k, uint32(0))))
				wtbCache[k] = (this.alpha + float32(sstable.DocTopic.Get(doc, k))) /
					(this.beta*float32(this.data.VocabSize) +
						float32(sstable.WordTopicSum.Get(k, uint32(0))))
			}

			for i, w := range corpus.ExpandWords(wcs) {
				// get the current topic of word w
				dw.DocId = doc
				dw.WordIdx = uint32(i)
				k := sstable.DocWordTopic[dw]

				// subtract old value from buckets
				denom := (this.beta*float32(this.data.VocabSize) +
					float32(sstable.WordTopicSum.Get(k, uint32(0))))
				smoothingBucket -= (this.alpha * this.beta) / denom
				docTopicBucket -= (this.beta * float32(sstable.DocTopic.Get(doc, k))) / denom

				// decrease corresponding sufficient statistics
				sstable.WordTopicMap.Decr(w, k, uint32(1))
				sstable.DocTopic.Decr(doc, k, uint32(1))
				sstable.WordTopicSum.Decr(k, uint32(0), uint32(1))

				// update bucket values
				denom = (this.beta*float32(this.data.VocabSize) +
					float32(sstable.WordTopicSum.Get(k, uint32(0))))
				smoothingBucket += (this.alpha * this.beta) / denom
				docTopicBucket += (this.beta * float32(sstable.DocTopic.Get(doc, k))) / denom
				wtbCache[k] = (this.alpha + float32(sstable.DocTopic.Get(doc, k))) / denom

				// compute word-topic bucket sum
				wtbSum := float32(0.0)
				for idx, _ := range sstable.WordTopicMap.Data[w] {
					tid, count := sstable.WordTopicMap.Get(w, idx)
					wtbSum += wtbCache[tid] * float32(count)
				}
				dtbSum := docTopicBucket
				sbSum := smoothingBucket

				// resample topic assignment
				var cumsum float32
				u := rand.Float32() * (wtbSum + dtbSum + sbSum)
				if u < wtbSum { // topic-word bucket
					cumsum = 0.0
					for tcIdx, _ := range sstable.WordTopicMap.Data[w] {
						tid, count := sstable.WordTopicMap.Get(w, tcIdx)
						cumsum += wtbCache[tid] * float32(count)
						if cumsum >= u {
							k = tid
							break
						}
					}
				} else if u < (wtbSum+dtbSum) && u >= wtbSum { // doc-topic bucket
					cumsum = 0.0
					u = u - wtbSum
					for kidx := uint32(0); kidx < this.topicNum; kidx += 1 {
						cumsum += (this.beta * float32(sstable.DocTopic.Get(doc, k))) / denom
						if cumsum >= u {
							k = kidx
							break
						}
					}
				} else { // smoothing bucket
					cumsum = 0.0
					for kidx := uint32(0); kidx < this.topicNum; kidx += 1 {
						cumsum += (this.alpha * this.beta) / denom
						if cumsum >= u {
							k = kidx
							break
						}
					}
				}

				denom = (this.beta*float32(this.data.VocabSize) +
					float32(sstable.WordTopicSum.Get(k, uint32(0))))
				smoothingBucket -= (this.alpha * this.beta) / denom
				docTopicBucket -= (this.beta * float32(sstable.DocTopic.Get(doc, k))) / denom

				// increase corresponding sufficient statistics
				sstable.WordTopicMap.Incr(w, k, uint32(1))
				sstable.DocTopic.Incr(doc, k, uint32(1))
				sstable.WordTopicSum.Incr(k, uint32(0), uint32(1))
				sstable.DocWordTopic[dw] = k

				// update bucket values
				denom = (this.beta*float32(this.data.VocabSize) +
					float32(sstable.WordTopicSum.Get(k, uint32(0))))
				smoothingBucket += (this.alpha * this.beta) / denom
				docTopicBucket += (this.beta * float32(sstable.DocTopic.Get(doc, k))) / denom
				wtbCache[k] = (this.alpha + float32(sstable.DocTopic.Get(doc, k))) / denom
			}
		}
	}
}

// infer topics on new documents
func (this *sparseLda) Infer(iter int) {}

// compute the posterior point estimation of word-topic mixture
// beta (Dirichlet prior) + data -> phi
func (this *sparseLda) Phi() *matrix.Float32Matrix {
	phi := matrix.NewFloat32Matrix(this.data.VocabSize, this.topicNum)

	for w := uint32(0); w < this.data.VocabSize; w += 1 {
		// convert sparse vector to dense vector
		wordTopicCount := make([]uint32, this.topicNum)
		for tcIdx, _ := range sstable.WordTopicMap.Data[w] {
			topicId, count := sstable.WordTopicMap.Get(w, tcIdx)
			wordTopicCount[topicId] = count
		}
		for k := uint32(0); k < this.topicNum; k += 1 {
			result := (float32(wordTopicCount[k]) + this.beta) /
				(float32(sstable.WordTopicSum.Get(k, uint32(0))) +
					float32(this.data.VocabSize)*this.beta)
			phi.Set(w, k, result)
		}
	}

	return phi
}

// compute the posterior point estimation of document-topic mixture
// alpha (Dirichlet prior) + data -> theta
func (this *sparseLda) Theta() *matrix.Float32Matrix {
	theta := matrix.NewFloat32Matrix(this.data.DocNum, this.topicNum)

	for d := uint32(0); d < this.data.DocNum; d += 1 {
		sum := util.Uint32VectorSum(sstable.DocTopic.GetRow(d))

		for k := uint32(0); k < this.topicNum; k += 1 {
			result := (float32(sstable.DocTopic.Get(d, k)) + this.alpha) /
				(float32(sum) + float32(this.topicNum)*this.alpha)
			theta.Set(d, k, result)
		}
	}

	return theta
}

// compute the joint likelihood of corpus
func (this *sparseLda) Likelihood() float64 {
	phi := this.Phi()
	theta := this.Theta()

	sum := float64(0.0)
	for doc, wcs := range this.data.Docs {
		for _, w := range corpus.ExpandWords(wcs) {
			topicSum := float32(0.0)
			for k := uint32(0); k < this.topicNum; k += 1 {
				topicSum += phi.Get(w, k) * theta.Get(doc, k)
			}
			sum += math.Log(float64(topicSum))
		}
	}

	return sum
}

// serialize word-topic distribution
func (this *sparseLda) SavePhi(fn string) error {
	phi := this.Phi()
	if err := phi.Serialize(fn + ".phi"); err != nil {
		return err
	}
	return nil
}

// serialize document-topic distribution
func (this *sparseLda) SaveTheta(fn string) error {
	theta := this.Theta()
	if err := theta.Serialize(fn + ".theta"); err != nil {
		return err
	}
	return nil
}

// serialize word-topic matrix
func (this *sparseLda) SaveWordTopic(fn string) error {
	if err := sstable.WordTopicMap.Serialize(fn + ".wt"); err != nil {
		return err
	}
	return nil
}

// deserialize word-topic matrix
func (this *sparseLda) LoadWordTopic(fn string) error {
	if err := sstable.WordTopicMap.Deserialize(fn + ".wt"); err != nil {
		return err
	}
	return nil
}
