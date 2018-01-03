package corpus

import (
	"bufio"
	"log"
	"os"
	"strconv"
	"strings"
)

type Corpus struct {
	VocabSize uint32
	DocNum    uint32
	Docs      map[uint32][]*WordCount
}

type WordCount struct {
	WordId uint32
	Count  uint32
}

func ExpandWords(wcs []*WordCount) []uint32 {
	var words []uint32
	for _, wc := range wcs {
		for i := uint32(0); i < wc.Count; i += 1 {
			words = append(words, wc.WordId)
		}
	}
	return words
}

// load training data from file, the file format should be like:
// [docId wordId:wordCount wordId:wordCount ... wordId:wordCount]
// the function will panic if docId, wordId and wordCount cannot
// be parsed to uint32
func (this *Corpus) Load(fn string) {
	f, err := os.Open(fn)
	if err != nil {
		panic(err)
	}

	if this.Docs == nil {
		this.Docs = make(map[uint32][]*WordCount)
	}
	vocabMaxId := uint32(0)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		doc := scanner.Text()
		vals := strings.Split(doc, " ")
		if len(vals) < 2 {
			log.Printf("bad document: %s", doc)
			continue
		}

		docId, err := strconv.ParseUint(vals[0], 10, 32)
		if err != nil {
			panic(err)
		}

		this.DocNum += uint32(1)

		for _, kv := range vals[1:] {
			wc := strings.Split(kv, ":")
			if len(wc) != 2 {
				log.Printf("bad word count: %s", kv)
				continue
			}

			wordId, err := strconv.ParseUint(wc[0], 10, 32)
			if err != nil {
				panic(err)
			}

			count, err := strconv.ParseUint(wc[1], 10, 32)
			if err != nil {
				panic(err)
			}

			this.Docs[uint32(docId)] = append(this.Docs[uint32(docId)], &WordCount{
				WordId: uint32(wordId),
				Count:  uint32(count),
			})
			if uint32(wordId) > vocabMaxId {
				vocabMaxId = uint32(wordId)
			}
		}
	}
	this.VocabSize = vocabMaxId + 1

	log.Printf("number of documents %d", this.DocNum)
	log.Printf("vocabulary size %d", this.VocabSize)
}
