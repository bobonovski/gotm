package model

import (
	"math/rand"
	"time"

	"github.com/bobonovski/gotm/corpus"
	"github.com/bobonovski/gotm/table"
)

type lda struct {
	data     *corpus.Corpus
	alpha    float32 // document topic mixture hyperparameter
	beta     float32 // topic word mixture hyperparameter
	topicNum uint32
}

// NewLDA creates a lda instance with collapsed gibbs sampler
func NewLDA(dat *corpus.Corpus,
	topicNum uint32, alpha float32, beta float32) *lda {
	// init sufficient statistics table
	table.Init(topicNum, dat.VocabSize, dat.DocNum)
	return &lda{
		data:     dat,
		alpha:    alpha,
		beta:     beta,
		topicNum: topicNum,
	}
}

func (this *lda) Run(iter int) {
	// randomly assign topic to word
	rand.Seed(time.Now().Unix())
	dw := table.DocWord{}
	for doc, wcs := range this.data.Docs {
		for i, w := range corpus.ExpandWords(wcs) {
			// sample word topic
			k := uint32(rand.Int31n(int32(this.topicNum)))

			// update sufficient statistics
			table.WordTopic.Incr(w, k, uint32(1))
			table.DocTopic.Incr(doc, k, uint32(1))
			table.WordTopicSum.Incr(k, uint32(1), uint32(1))

			// update doc word topic assignment
			dw.DocId = doc
			dw.WordIdx = uint32(i)
			table.DocWordTopic[dw] = k
		}
	}
	// collapsed gibbs sampling
	cumsum := make([]float32, this.topicNum)
	for doc, wcs := range this.data.Docs {
		for i, w := range corpus.ExpandWords(wcs) {
			// get the current topic of word w
			dw.DocId = doc
			dw.WordIdx = uint32(i)
			k := table.DocWordTopic[dw]
			// decrease corresponding sufficient statistics
			table.WordTopic.Decr(w, k, uint32(1))
			table.DocTopic.Decr(doc, k, uint32(1))
			table.WordTopicSum.Decr(k, uint32(1), uint32(1))
			// resample the topic
			for kidx := uint32(0); kidx < this.topicNum; kidx += 1 {
				docPart := this.alpha + float32(table.DocTopic.Get(doc, k))
				wordPart := (this.beta + float32(table.WordTopic.Get(w, k))) /
					(float32(table.WordTopicSum.Get(k, uint32(1))) +
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
			table.WordTopic.Incr(w, k, uint32(1))
			table.DocTopic.Incr(doc, k, uint32(1))
			table.WordTopicSum.Incr(k, uint32(1), uint32(1))
			table.DocWordTopic[dw] = k
		}
	}
}
