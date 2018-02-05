package sstable

// uint32 vector summation
func Uint32VectorSum(data []uint32) uint32 {
	sum := uint32(0)
	for _, d := range data {
		sum += d
	}
	return sum
}

// float32 vector summation
func Float32VectorSum(data []float32) float32 {
	sum := float32(0.0)
	for _, d := range data {
		sum += d
	}
	return sum
}
