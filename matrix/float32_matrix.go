package matrix

// internal Float32 matrix representation
type Float32Matrix struct {
	nrow uint32
	ncol uint32
	data []float32
}

// NewFloat32Matrix creates a new Float32Matrix with r rows and c columns
// which is mainly used for caching temporary results
func NewFloat32Matrix(r, c uint32) *Float32Matrix {
	if r*c <= 0 {
		panic(ErrIndexOutOfRange)
	}
	return &Float32Matrix{
		nrow: r,
		ncol: c,
		data: make([]float32, r*c),
	}
}

// get the shape of the matrix
func (m *Float32Matrix) Shape() (uint32, uint32) {
	return m.nrow, m.ncol
}

// get the [r, c]-th element of the matrix
func (m *Float32Matrix) Get(r, c uint32) float32 {
	if r >= m.nrow || c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}
	return m.data[r*m.ncol+c]
}

// set val to the [r, c]-th element of the matrix
func (m *Float32Matrix) Set(r, c uint32, val float32) {
	if r >= m.nrow || c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}
	m.data[r*m.ncol+c] = val
}
