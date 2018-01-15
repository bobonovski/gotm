package model

type Model interface {
	Run(iter int)
	SaveModel(fn string) error
}
