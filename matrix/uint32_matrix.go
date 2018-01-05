package matrix

// internal Uint32 matrix representation
type Uint32Matrix struct {
	nrow uint32
	ncol uint32
	data []uint32
}

// NewUint32Matrix creates a new Uint32Matrix with r rows and c columns.
// if r*c <= 0, it will panic. A uint32 slice is used as the underlying
// storage and the data layout is in row major order, i.e. the (i*c + j)-th
// element in the data slice is the [i, j]-th element in the matrix.
// Vector is defined as a matrix one column, i.e. a column vector.
func NewUint32Matrix(r, c uint32) *Uint32Matrix {
	if r*c <= 0 {
		panic(ErrIndexOutOfRange)
	}
	return &Uint32Matrix{
		nrow: r,
		ncol: c,
		data: make([]uint32, r*c),
	}
}

// get the shape of the matrix
func (m *Uint32Matrix) Shape() (uint32, uint32) {
	return m.nrow, m.ncol
}

// get the [r, c]-th element of the matrix
func (m *Uint32Matrix) Get(r, c uint32) uint32 {
	if r >= m.nrow || c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}
	return m.data[r*m.ncol+c]
}

// get the r-th row of the matrix
func (m *Uint32Matrix) GetRow(r uint32) []uint32 {
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
func (m *Uint32Matrix) GetCol(c uint32) []uint32 {
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
func (m *Uint32Matrix) Set(r, c uint32, val uint32) {
	if r >= m.nrow || c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}
	m.data[r*m.ncol+c] = val
}

// increment the [r, c]-th element of the matrix by val
func (m *Uint32Matrix) Incr(r, c uint32, val uint32) {
	if r >= m.nrow || c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}
	m.data[r*m.ncol+c] += val
}

// decrement the [r, c]-th element of the matrix by val
func (m *Uint32Matrix) Decr(r, c uint32, val uint32) {
	if r >= m.nrow || c >= m.ncol {
		panic(ErrIndexOutOfRange)
	}
	m.data[r*m.ncol+c] -= val
}
