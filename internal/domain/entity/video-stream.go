package entity

type VideoStream struct {
	ID    string      `json:"id"`
	Label string      `json:"label"`
	Input interface{} `json:"input"`
}

func NewVideoStream(id, label string, input interface{}) *VideoStream {
	return &VideoStream{
		ID:    id,
		Label: label,
		Input: input,
	}
}
