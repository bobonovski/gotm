package table

import "github.com/bobonovski/gotm/matrix"

var (
	// [w, t]-th element counts how many times
	// word w has been assigned to topic t
	WordTopic *matrix.Uint32Matrix
	// [d, t]-th element counts how many words
	// in d has been assigned to topic t
	DocTopic *matrix.Uint32Matrix
	// vector of length topicNum: [t]-th element counts
	// how many words in total has been assigned to topic t
	WordTopicSum *matrix.Uint32Matrix
	// hashmap which remembers the topic of i-th word of doc d
	// has been assigned before
	DocWordTopic map[DocWord]uint32
)

type DocWord struct {
	DocId   uint32
	WordIdx uint32
}
