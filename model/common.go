package model

import (
	"fmt"

	"github.com/bobonovski/gotm/corpus"
)

var constructors = make(map[string]ModelCtor)

// the common interface new LDA samplers should follow
type Model interface {
	// train model for iter iteration
	Train(iter int)
	// do inference for new doc for iter iteration
	Infer(iter int)
	// serialize posterior document topic distribution
	SaveTheta(fn string) error
	// serialize posterior word topic distribution
	SavePhi(fn string) error
	// serialize word topic count table
	SaveWordTopic(fn string) error
	// deserialize word topic count table
	LoadWordTopic(fn string) error
	// called when the model should be created
}

// new LDA sampler should register itself using this function
func Register(modelType string, m ModelCtor) {
	constructors[modelType] = m
}

type ModelCtor func(dat *corpus.Corpus, topicNum uint32, alpha float32, beta float32) Model

func GetModel(modelType string) (ModelCtor, error) {
	if _, ok := constructors[modelType]; !ok {
		return nil, fmt.Errorf("model %s not registered", modelType)
	}
	return constructors[modelType], nil
}
