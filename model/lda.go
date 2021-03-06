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
	Alpha    float32 // document topic mixture hyperparameter
	Beta     float32 // topic word mixture hyperparameter
	TopicNum uint32

	Data *corpus.Corpus // for convenience

	Wt  *sstable.Uint32Matrix      // word-topic count table
	Dt  *sstable.Uint32Matrix      // doc-topic count table
	Wts *sstable.Uint32Matrix      // word-topic-sum count table
	Dwt map[sstable.DocWord]uint32 // doc-word-topic map
}

// New creates a LDA instance with collapsed gibbs sampler
func NewLDA(topicNum uint32, alpha float32, beta float32) Model {
	return &LDA{
		Alpha:    alpha,
		Beta:     beta,
		TopicNum: topicNum,
	}
}

func (this *LDA) Init() {
	// randomly assign topic to word
	rand.Seed(time.Now().Unix())
	dw := sstable.DocWord{}
	for doc, wcs := range this.Data.Docs {
		for i, w := range corpus.ExpandWords(wcs) {
			// sample word topic
			k := uint32(rand.Int31n(int32(this.TopicNum)))

			// update sufficient statistics
			this.Wt.Incr(w, k, uint32(1))
			this.Dt.Incr(doc, k, uint32(1))
			this.Wts.Incr(k, uint32(0), uint32(1))

			// update doc word topic assignment
			dw.DocId = doc
			dw.WordIdx = uint32(i)
			this.Dwt[dw] = k
		}
	}
}

func (this *LDA) ResampleTopics(iter int) {
	dw := sstable.DocWord{}
	cumsum := make([]float32, this.TopicNum)

	for iterIdx := 0; iterIdx < iter; iterIdx += 1 {
		if log.V(5) {
			if iterIdx%10 == 0 {
				log.Infof("iter %5d, likelihood %f", iterIdx, this.Likelihood())
			}
		}
		// collapsed gibbs sampling
		for doc, wcs := range this.Data.Docs {
			for i, w := range corpus.ExpandWords(wcs) {
				// get the current topic of word w
				dw.DocId = doc
				dw.WordIdx = uint32(i)
				k := this.Dwt[dw]

				// decrease corresponding sufficient statistics
				this.Wt.Decr(w, k, uint32(1))
				this.Dt.Decr(doc, k, uint32(1))
				this.Wts.Decr(k, uint32(0), uint32(1))

				// resample the topic
				for kidx := uint32(0); kidx < this.TopicNum; kidx += 1 {
					docPart := this.Alpha + float32(this.Dt.Get(doc, kidx))
					wordPart := (this.Beta + float32(this.Wt.Get(w, kidx))) /
						(float32(this.Wts.Get(kidx, uint32(0))) +
							this.Beta*float32(this.Data.VocabSize))
					if kidx == 0 {
						cumsum[kidx] = docPart * wordPart
					} else {
						cumsum[kidx] = cumsum[kidx-1] + docPart*wordPart
					}
				}
				u := rand.Float32() * cumsum[this.TopicNum-1]
				for kidx := uint32(0); kidx < this.TopicNum; kidx += 1 {
					if u < cumsum[kidx] {
						k = kidx
						break
					}
				}

				// increase corresponding sufficient statistics
				this.Wt.Incr(w, k, uint32(1))
				this.Dt.Incr(doc, k, uint32(1))
				this.Wts.Incr(k, uint32(0), uint32(1))
				this.Dwt[dw] = k
			}
		}
	}
}

func (this *LDA) Train(dat *corpus.Corpus, iter int) {
	if dat == nil {
		log.Fatal("corpus is nil")
	}
	// create sstables
	this.Wt = sstable.NewUint32Matrix(dat.VocabSize, this.TopicNum)
	this.Dt = sstable.NewUint32Matrix(dat.DocNum, this.TopicNum)
	this.Wts = sstable.NewUint32Matrix(this.TopicNum, uint32(1))
	this.Dwt = make(map[sstable.DocWord]uint32)
	this.Data = dat

	// randomly init sstables
	this.Init()

	this.ResampleTopics(iter)
}

// infer topics on new documents
func (this *LDA) Infer(dat *corpus.Corpus, iter int) {
	if dat == nil {
		log.Fatal("corpus is nil")
	}
	if this.Wt == nil || this.Wts == nil {
		log.Fatal("Wt or Wts is not initialized, maybe model is not loaded")
	}
	// Wt, Wts should be initialized when model was loaded
	this.Dt = sstable.NewUint32Matrix(dat.DocNum, this.TopicNum)
	this.Dwt = make(map[sstable.DocWord]uint32)
	this.Data = dat

	// randomly init sstables
	this.Init()

	this.ResampleTopics(iter)
}

// compute the posterior point estimation of word-topic mixture
// beta (Dirichlet prior) + data -> phi
func (this *LDA) Phi() *sstable.Float32Matrix {
	phi := sstable.NewFloat32Matrix(this.Data.VocabSize, this.TopicNum)

	for k := uint32(0); k < this.TopicNum; k += 1 {
		sum := sstable.Uint32VectorSum(this.Wt.GetCol(k))

		for v := uint32(0); v < this.Data.VocabSize; v += 1 {
			result := (float32(this.Wt.Get(v, k)) + this.Beta) /
				(float32(sum) + float32(this.Data.VocabSize)*this.Beta)
			phi.Set(v, k, result)
		}
	}

	return phi
}

// compute the posterior point estimation of document-topic mixture
// alpha (Dirichlet prior) + data -> theta
func (this *LDA) Theta() *sstable.Float32Matrix {
	theta := sstable.NewFloat32Matrix(this.Data.DocNum, this.TopicNum)

	for d := uint32(0); d < this.Data.DocNum; d += 1 {
		sum := sstable.Uint32VectorSum(this.Dt.GetRow(d))

		for k := uint32(0); k < this.TopicNum; k += 1 {
			result := (float32(this.Dt.Get(d, k)) + this.Alpha) /
				(float32(sum) + float32(this.TopicNum)*this.Alpha)
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

// serialize word-topic distribution
func (this *LDA) SavePhi(fn string) error {
	phi := this.Phi()
	if err := sstable.Float32Serialize(phi, fn); err != nil {
		return err
	}
	return nil
}

// serialize document-topic distribution
func (this *LDA) SaveTheta(fn string) error {
	theta := this.Theta()
	if err := sstable.Float32Serialize(theta, fn); err != nil {
		return err
	}
	return nil
}

// serialize word-topic matrix
func (this *LDA) SaveWordTopic(fn string) error {
	if err := sstable.Uint32Serialize(this.Wt, fn); err != nil {
		return err
	}
	return nil
}

// deserialize word-topic matrix
func (this *LDA) LoadWordTopic(fn string) error {
	v, err := sstable.Uint32Deserialize(fn)
	if err != nil {
		return err
	}
	this.Wt = v
	// init WordTopicSum table
	this.Wts = sstable.NewUint32Matrix(this.TopicNum, uint32(1))
	vocab, topicNum := this.Wt.Shape()
	log.Infof("%d,%d", vocab, topicNum)
	for r := uint32(0); r < vocab; r += 1 {
		for t := uint32(0); t < topicNum; t += 1 {
			this.Wts.Incr(t, uint32(0), this.Wt.Get(r, t))
		}
	}
	return nil
}
