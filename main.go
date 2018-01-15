package main

import (
	"flag"
	"log"

	"github.com/bobonovski/gotm/corpus"
	"github.com/bobonovski/gotm/model"
)

var (
	input     = flag.String("input_file", "", "input training file")
	modelType = flag.String("model_type", "lda", "model type")
	alpha     = flag.Float64("alpha", 0.01, "document-topic mixture hyperparameter")
	beta      = flag.Float64("beta", 0.01, "topic-word mixture hyperparameter")
	topicNum  = flag.Uint("k", 20, "number of topics")
	iteration = flag.Int("iter", 10, "number of iteration")
	modelName = flag.String("model_file", "lda_model", "input/output model name")
)

func main() {
	flag.Parse()

	// read training data
	data := &corpus.Corpus{}
	data.Load(*input)

	// init model
	var m model.Model
	switch *modelType {
	case "lda":
		m = model.NewLDA(data, uint32(*topicNum), float32(*alpha), float32(*beta))
	case "sparselda":
		m = model.NewSparseLDA(data, uint32(*topicNum), float32(*alpha), float32(*beta))
	default:
		log.Printf("not supported yet")
	}

	// run model
	m.Run(*iteration)

	// save model
	m.SaveModel(*modelName)
}
