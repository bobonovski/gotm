package model

type Model interface {
	Train(iter int)
	Infer(iter int)
	SaveTheta(fn string) error
	SavePhi(fn string) error
	SaveWordTopic(fn string) error
	LoadWordTopic(fn string) error
}
