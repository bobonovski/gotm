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

// get the [r, c]-th element of the matrix
func (m *denseMatrix) Get(r, c uint32) uint32 {
	if r >= m.nrow || c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}
	return m.data[r*m.ncol+c]
}

// get the r-th row of the matrix
func (m *denseMatrix) GetRow(r uint32) []uint32 {
	if r >= m.nrow {
		panic(ErrIndexOutOfRange)
	}

	var row []uint32
	for c := uint32(0); c < m.ncol; c += 1 {
		row = append(row, m.Get(r, c))
	}
	return row
}

// get the c-th column of the matrix
func (m *denseMatrix) GetCol(c uint32) []uint32 {
	if c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}

	var column []uint32
	for r := uint32(0); r < m.nrow; r += 1 {
		column = append(column, m.Get(r, c))
	}
	return column
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
