package main

import (
	"flag"

	log "github.com/golang/glog"

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
	infer     = flag.Bool("infer", false, "whether do inference on input file")
)

func main() {
	flag.Parse()

	// load documents for training or inference
	data := &corpus.Corpus{}
	data.Load(*input)

	// init model
	ctor, err := model.GetModel(*modelType)
	if err != nil {
		log.Fatal(err)
	}
	m := ctor(uint32(*topicNum), float32(*alpha), float32(*beta))

	if *infer == false {
		log.Infof("training for new %s model", *modelType)
		// train model
		m.Train(data, *iteration)
		// save document-topic distribution
		m.SaveTheta(*modelName)
		// save word-topic distribution
		m.SavePhi(*modelName)
		// save word-topic matrix
		m.SaveWordTopic(*modelName)
	} else {
		log.Infof("infer for new docs")
		// load word-topic matrix
		m.LoadWordTopic(*modelName)
		// infer document topics
		m.Infer(data, *iteration)
		// save document-topic distribution
		m.SaveTheta(*modelName)
	}
}
