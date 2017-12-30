package matrix

import "errors"

var (
	ErrIndexOutOfRange = errors.New("matrix: index out of range")
	ErrBadShape        = errors.New("matrix: non-positive dimension not allowed")
)
