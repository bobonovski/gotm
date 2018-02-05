package model

import (
	"math"
	"math/rand"
	"time"

	log "github.com/golang/glog"

	"github.com/bobonovski/gotm/corpus"
	"github.com/bobonovski/gotm/sstable"
)

func init() {
	Register("lda", NewLDA)
}

type LDA struct {
	data     *corpus.Corpus
	alpha    float32 // document topic mixture hyperparameter
	beta     float32 // topic word mixture hyperparameter
	topicNum uint32

	wt  *sstable.Uint32Matrix      // word-topic count table
	dt  *sstable.Uint32Matrix      // doc-topic count table
	wts *sstable.Uint32Matrix      // word-topic-sum count table
	dwt map[sstable.DocWord]uint32 // doc-word-topic map
}

// New creates a LDA instance with collapsed gibbs sampler
func NewLDA(dat *corpus.Corpus,
	topicNum uint32, alpha float32, beta float32) Model {
	return &LDA{
		data:     dat,
		alpha:    alpha,
		beta:     beta,
		topicNum: topicNum,
		wt:       sstable.NewUint32Matrix(dat.VocabSize, topicNum),
		dt:       sstable.NewUint32Matrix(dat.DocNum, topicNum),
		wts:      sstable.NewUint32Matrix(topicNum, uint32(1)),
		dwt:      make(map[sstable.DocWord]uint32),
	}
}

func (this *LDA) Init() {
	// randomly assign topic to word
	rand.Seed(time.Now().Unix())
	dw := sstable.DocWord{}
	for doc, wcs := range this.data.Docs {
		for i, w := range corpus.ExpandWords(wcs) {
			// sample word topic
			k := uint32(rand.Int31n(int32(this.topicNum)))

			// update sufficient statistics
			this.wt.Incr(w, k, uint32(1))
			this.dt.Incr(doc, k, uint32(1))
			this.wts.Incr(k, uint32(0), uint32(1))

			// update doc word topic assignment
			dw.DocId = doc
			dw.WordIdx = uint32(i)
			this.dwt[dw] = k
		}
	}
}

func (this *LDA) Train(iter int) {
	this.Init()
	dw := sstable.DocWord{}
	for iterIdx := 0; iterIdx < iter; iterIdx += 1 {
		if iterIdx%10 == 0 {
			log.Infof("iter %5d, likelihood %f", iterIdx, this.Likelihood())
		}

		// collapsed gibbs sampling
		cumsum := make([]float32, this.topicNum)
		for doc, wcs := range this.data.Docs {
			for i, w := range corpus.ExpandWords(wcs) {
				// get the current topic of word w
				dw.DocId = doc
				dw.WordIdx = uint32(i)
				k := this.dwt[dw]

				// decrease corresponding sufficient statistics
				this.wt.Decr(w, k, uint32(1))
				this.dt.Decr(doc, k, uint32(1))
				this.wts.Decr(k, uint32(0), uint32(1))

				// resample the topic
				for kidx := uint32(0); kidx < this.topicNum; kidx += 1 {
					docPart := this.alpha + float32(this.dt.Get(doc, kidx))
					wordPart := (this.beta + float32(this.wt.Get(w, kidx))) /
						(float32(this.wts.Get(kidx, uint32(0))) +
							this.beta*float32(this.data.VocabSize))
					if kidx == 0 {
						cumsum[kidx] = docPart * wordPart
					} else {
						cumsum[kidx] = cumsum[kidx-1] + docPart*wordPart
					}
				}
				u := rand.Float32() * cumsum[this.topicNum-1]
				for kidx := uint32(0); kidx < this.topicNum; kidx += 1 {
					if u < cumsum[kidx] {
						k = kidx
						break
					}
				}

				// increase corresponding sufficient statistics
				this.wt.Incr(w, k, uint32(1))
				this.dt.Incr(doc, k, uint32(1))
				this.wts.Incr(k, uint32(0), uint32(1))
				this.dwt[dw] = k
			}
		}
	}
}

// infer topics on new documents
func (this *LDA) Infer(iter int) {
	this.Train(iter)
}

// compute the posterior point estimation of word-topic mixture
// beta (Dirichlet prior) + data -> phi
func (this *LDA) Phi() *sstable.Float32Matrix {
	phi := sstable.NewFloat32Matrix(this.data.VocabSize, this.topicNum)

	for k := uint32(0); k < this.topicNum; k += 1 {
		sum := sstable.Uint32VectorSum(this.wt.GetCol(k))

		for v := uint32(0); v < this.data.VocabSize; v += 1 {
			result := (float32(this.wt.Get(v, k)) + this.beta) /
				(float32(sum) + float32(this.data.VocabSize)*this.beta)
			phi.Set(v, k, result)
		}
	}

	return phi
}

// compute the posterior point estimation of document-topic mixture
// alpha (Dirichlet prior) + data -> theta
func (this *LDA) Theta() *sstable.Float32Matrix {
	theta := sstable.NewFloat32Matrix(this.data.DocNum, this.topicNum)

	for d := uint32(0); d < this.data.DocNum; d += 1 {
		sum := sstable.Uint32VectorSum(this.dt.GetRow(d))

		for k := uint32(0); k < this.topicNum; k += 1 {
			result := (float32(this.dt.Get(d, k)) + this.alpha) /
				(float32(sum) + float32(this.topicNum)*this.alpha)
			theta.Set(d, k, result)
		}
	}

	return theta
}

// compute the joint likelihood of corpus
func (this *LDA) Likelihood() float64 {
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
func (this *LDA) SavePhi(fn string) error {
	phi := this.Phi()
	if err := phi.Serialize(fn + ".phi"); err != nil {
		return err
	}
	return nil
}

// serialize document-topic distribution
func (this *LDA) SaveTheta(fn string) error {
	theta := this.Theta()
	if err := theta.Serialize(fn + ".theta"); err != nil {
		return err
	}
	return nil
}

// serialize word-topic matrix
func (this *LDA) SaveWordTopic(fn string) error {
	if err := this.wt.Serialize(fn + ".wt"); err != nil {
		return err
	}
	return nil
}

// deserialize word-topic matrix
func (this *LDA) LoadWordTopic(fn string) error {
	if err := this.wt.Deserialize(fn + ".wt"); err != nil {
		return err
	}
	return nil
}
