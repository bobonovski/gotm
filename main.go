package main

import (
	"flag"
	"log"

	"github.com/bobonovski/gotm/corpus"
	"github.com/bobonovski/gotm/model"
)

var (
	input      = flag.String("input_file", "", "input training file")
	topicModel = flag.String("model", "lda", "model type")
	alpha      = flag.Float64("alpha", 0.01, "document-topic mixture hyperparameter")
	beta       = flag.Float64("beta", 0.01, "topic-word mixture hyperparameter")
	topicNum   = flag.Uint("k", 20, "number of topics")
	iteration  = flag.Int("iter", 10, "number of iteration")
)

func main() {
	flag.Parse()

	// read training data
	data := &corpus.Corpus{}
	data.Load(*input)

	// init model
	var m model.Model
	switch *topicModel {
	case "lda":
		m = model.NewLDA(data, uint32(*topicNum), float32(*alpha), float32(*beta))
	case "sparselda":
		m = model.NewSparseLDA(data, uint32(*topicNum), float32(*alpha), float32(*beta))
	default:
		log.Printf("not supported yet")
	}

	m.Run(*iteration)
}
