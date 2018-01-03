package matrix

// internal Cache matrix representation
type CacheMatrix struct {
	nrow uint32
	ncol uint32
	data []float32
}

// NewCacheMatrix creates a new CacheMatrix with r rows and c columns
// which is mainly used for caching temporary results
func NewCacheMatrix(r, c uint32) *CacheMatrix {
	if r*c <= 0 {
		panic(ErrIndexOutOfRange)
	}
	return &CacheMatrix{
		nrow: r,
		ncol: c,
		data: make([]float32, r*c),
	}
}

// get the shape of the matrix
func (m *CacheMatrix) Shape() (uint32, uint32) {
	return m.nrow, m.ncol
}

// get the [r, c]-th element of the matrix
func (m *CacheMatrix) Get(r, c uint32) float32 {
	if r >= m.nrow || c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}
	return m.data[r*m.ncol+c]
}

// set val to the [r, c]-th element of the matrix
func (m *CacheMatrix) Set(r, c uint32, val float32) {
	if r >= m.nrow || c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}
	m.data[r*m.ncol+c] = val
}
