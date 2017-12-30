package matrix

// internal dense matrix representation
type denseMatrix struct {
	nrow uint32
	ncol uint32
	data []uint32
}

// NewDenseMatrix creates a new denseMatrix with r rows and c columns.
// if r*c <= 0, it will panic. A uint32 slice is used as the underlying
// storage and the data layout is in row major order, i.e. the (i*c + j)-th
// element in the data slice is the [i, j]-th element in the matrix.
// Vector is defined as a matrix one column, i.e. a column vector.
func NewDenseMatrix(r, c uint32) *denseMatrix {
	if r*c <= 0 {
		panic(ErrIndexOutOfRange)
	}
	return &denseMatrix{
		nrow: r,
		ncol: c,
		data: make([]uint32, r*c),
	}
}

// get the shape of the matrix
func (m *denseMatrix) Shape() (uint32, uint32) {
	return m.nrow, m.ncol
}

// get the [r, c]-the element of the matrix
func (m *denseMatrix) Get(r, c uint32) uint32 {
	if r >= m.nrow || c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}
	return m.data[r*m.ncol+c]
}

// set val to the [r, c]-th element of the matrix
func (m *denseMatrix) Set(r, c uint32, val uint32) {
	if r >= m.nrow || c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}
	m.data[r*m.ncol+c] = val
}

// increment the [r, c]-th element of the matrix by val
func (m *denseMatrix) Incr(r, c uint32, val uint32) {
	if r >= m.nrow || c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}
	m.data[r*m.ncol+c] += val
}

// decrement the [r, c]-th element of the matrix by val
func (m *denseMatrix) Decr(r, c uint32, val uint32) {
	if r >= m.nrow || c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}
	m.data[r*m.ncol+c] -= val
}
