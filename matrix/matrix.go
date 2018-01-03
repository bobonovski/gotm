package matrix

type Matrix interface {
	Shape() (uint32, uint32)
	Get(uint32, uint32) uint32
	Set(uint32, uint32, uint32)
	Incr(uint32, uint32, uint32)
	Decr(uint32, uint32, uint32)
	GetRow(uint32) []uint32
	GetCol(uint32) []uint32
}
