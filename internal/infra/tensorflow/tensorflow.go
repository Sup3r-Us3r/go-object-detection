package tensorflow

import (
	"bytes"
	"fmt"
	"image"
	"log"

	tf "github.com/wamuir/graft/tensorflow"
	"github.com/wamuir/graft/tensorflow/op"
)

type MachineLearningInterface interface {
	LoadSavedModel(modelPath string) (model *tf.SavedModel, tfSession *tf.Session, err error)
	DecodeBitmapGraph() (tfGraph *tf.Graph, input, output tf.Output, err error)
	MakeTensorFromImage(imageTarget []byte) (tensor *tf.Tensor, imageDecode image.Image, err error)
	PredictObjectBoxes(input *tf.Tensor) (probabilities, classes []float32, boxes [][]float32, err error)
}

var COCO_SSD_LABELS = map[int]string{
	0:   "unlabeled",
	1:   "person",
	2:   "bicycle",
	3:   "car",
	4:   "motorcycle",
	5:   "airplane",
	6:   "bus",
	7:   "train",
	8:   "truck",
	9:   "boat",
	10:  "traffic light",
	11:  "fire hydrant",
	12:  "street sign",
	13:  "stop sign",
	14:  "parking meter",
	15:  "bench",
	16:  "bird",
	17:  "cat",
	18:  "dog",
	19:  "horse",
	20:  "sheep",
	21:  "cow",
	22:  "elephant",
	23:  "bear",
	24:  "zebra",
	25:  "giraffe",
	26:  "hat",
	27:  "backpack",
	28:  "umbrella",
	29:  "shoe",
	30:  "eye glasses",
	31:  "handbag",
	32:  "tie",
	33:  "suitcase",
	34:  "frisbee",
	35:  "skis",
	36:  "snowboard",
	37:  "sports ball",
	38:  "kite",
	39:  "baseball bat",
	40:  "baseball glove",
	41:  "skateboard",
	42:  "surfboard",
	43:  "tennis racket",
	44:  "bottle",
	45:  "plate",
	46:  "wine glass",
	47:  "cup",
	48:  "fork",
	49:  "knife",
	50:  "spoon",
	51:  "bowl",
	52:  "banana",
	53:  "apple",
	54:  "sandwich",
	55:  "orange",
	56:  "broccoli",
	57:  "carrot",
	58:  "hot dog",
	59:  "pizza",
	60:  "donut",
	61:  "cake",
	62:  "chair",
	63:  "couch",
	64:  "potted plant",
	65:  "bed",
	66:  "mirror",
	67:  "dining table",
	68:  "window",
	69:  "desk",
	70:  "toilet",
	71:  "door",
	72:  "tv",
	73:  "laptop",
	74:  "mouse",
	75:  "remote",
	76:  "keyboard",
	77:  "cell phone",
	78:  "microwave",
	79:  "oven",
	80:  "toaster",
	81:  "sink",
	82:  "refrigerator",
	83:  "blender",
	84:  "book",
	85:  "clock",
	86:  "vase",
	87:  "scissors",
	88:  "teddy bear",
	89:  "hair drier",
	90:  "toothbrush",
	91:  "hair brush",
	92:  "banner",
	93:  "blanket",
	94:  "branch",
	95:  "bridge",
	96:  "building-other",
	97:  "bush",
	98:  "cabinet",
	99:  "cage",
	100: "cardboard",
	101: "carpet",
	102: "ceiling-other",
	103: "ceiling-tile",
	104: "cloth",
	105: "clothes",
	106: "clouds",
	107: "counter",
	108: "cupboard",
	109: "curtain",
	110: "desk-stuff",
	111: "dirt",
	112: "door-stuff",
	113: "fence",
	114: "floor-marble",
	115: "floor-other",
	116: "floor-stone",
	117: "floor-tile",
	118: "floor-wood",
	119: "flower",
	120: "fog",
	121: "food-other",
	122: "fruit",
	123: "furniture-other",
	124: "grass",
	125: "gravel",
	126: "ground-other",
	127: "hill",
	128: "house",
	129: "leaves",
	130: "light",
	131: "mat",
	132: "metal",
	133: "mirror-stuff",
	134: "moss",
	135: "mountain",
	136: "mud",
	137: "napkin",
	138: "net",
	139: "paper",
	140: "pavement",
	141: "pillow",
	142: "plant-other",
	143: "plastic",
	144: "platform",
	145: "playingfield",
	146: "railing",
	147: "railroad",
	148: "river",
	149: "road",
	150: "rock",
	151: "roof",
	152: "rug",
	153: "salad",
	154: "sand",
	155: "sea",
	156: "shelf",
	157: "sky-other",
	158: "skyscraper",
	159: "snow",
	160: "solid-other",
	161: "stairs",
	162: "stone",
	163: "straw",
	164: "structural-other",
	165: "table",
	166: "tent",
	167: "textile-other",
	168: "towel",
	169: "tree",
	170: "vegetable",
	171: "wall-brick",
	172: "wall-concrete",
	173: "wall-other",
	174: "wall-panel",
	175: "wall-stone",
	176: "wall-tile",
	177: "wall-wood",
	178: "water-other",
	179: "waterdrops",
	180: "window-blind",
	181: "window-other",
	182: "wood",
}

var (
	// TF session, re-usable and concurrency safe
	session *tf.Session
	// Model graph
	graph *tf.Graph
)

type TensorflowMachineLearning struct{}

func NewTensorflowMachineLearning() *TensorflowMachineLearning {
	return &TensorflowMachineLearning{}
}

// Load pre-trained model from COCO SSD
func (tml *TensorflowMachineLearning) LoadSavedModel(modelPath string) (model *tf.SavedModel, tfSession *tf.Session, err error) {
	model, err = tf.LoadSavedModel(modelPath, []string{"serve"}, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("unable to load saved model: %v", err)
	}

	graph = model.Graph

	// Create a session for inference over graph
	session, err = tf.NewSession(graph, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("unable to create session: %v", err)
	}

	return model, session, nil
}

// Build a graph to decode bitmap input into the proper tensor shape
// The object detection models take an input of [1, ?, ?, 3]
func (tml *TensorflowMachineLearning) DecodeBitmapGraph() (tfGraph *tf.Graph, input, output tf.Output, err error) {
	scope := op.NewScope()

	input = op.Placeholder(scope, tf.String)

	output = op.ExpandDims(
		scope,
		op.DecodeBmp(scope, input, op.DecodeBmpChannels(3)),
		op.Const(scope.SubScope("make_batch"), int32(0)),
	)

	tfGraph, err = scope.Finalize()

	return
}

// Create a tensor from an image bytes
func (tml *TensorflowMachineLearning) MakeTensorFromImage(imageTarget []byte) (tensor *tf.Tensor, imageDecode image.Image, err error) {
	// DecodeJpeg uses a scalar String-valued tensor as input
	tensor, err = tf.NewTensor(string(imageTarget))
	if err != nil {
		return nil, nil, err
	}

	// Creates a tensorflow graph to decode the jpeg image
	tfGraph, input, output, err := tml.DecodeBitmapGraph()
	if err != nil {
		return nil, nil, err
	}

	// Execute that graph to decode this one image
	tfSession, err := tf.NewSession(tfGraph, nil)
	if err != nil {
		return nil, nil, err
	}
	defer tfSession.Close()

	normalized, err := tfSession.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil,
	)
	if err != nil {
		return nil, nil, err
	}

	imageReader := bytes.NewReader(imageTarget)
	imageDecode, _, err = image.Decode(imageReader)
	if err != nil {
		return nil, nil, err
	}

	return normalized[0], imageDecode, nil
}

// Returns the probabilities, classes and boxes located in the processed frame
func (tml *TensorflowMachineLearning) PredictObjectBoxes(input *tf.Tensor) (probabilities, classes []float32, boxes [][]float32, err error) {
	// Get all the input and output operations
	inputOp := graph.Operation("image_tensor")

	// Output ops
	o1 := graph.Operation("detection_boxes")
	o2 := graph.Operation("detection_scores")
	o3 := graph.Operation("detection_classes")
	o4 := graph.Operation("num_detections")

	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			inputOp.Output(0): input,
		},
		[]tf.Output{
			o1.Output(0),
			o2.Output(0),
			o3.Output(0),
			o4.Output(0),
		},
		nil,
	)
	if err != nil {
		log.Fatalf("error running session: %v", err)
	}

	boxes = output[0].Value().([][][]float32)[0]
	probabilities = output[1].Value().([][]float32)[0]
	classes = output[2].Value().([][]float32)[0]

	return
}
