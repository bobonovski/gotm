package util

// sum the vector
func VectorSum(data []uint32) uint32 {
	sum := uint32(0)
	for _, d := range data {
		sum += d
	}
	return sum
}

func Float32VectorSum(data []float32) float32 {
	sum := float32(0.0)
	for _, d := range data {
		sum += d
	}
	return sum
}
