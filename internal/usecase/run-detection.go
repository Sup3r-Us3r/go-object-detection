package usecase

import (
	"fmt"
	"image"
	"log"

	"gocv.io/x/gocv"

	"github.com/Sup3r-Us3r/go-object-detection/internal/infra/opencv"
	"github.com/Sup3r-Us3r/go-object-detection/internal/infra/tensorflow"
)

type RunDetectionUseCase struct {
	Tf     tensorflow.MachineLearningInterface
	OpenCV opencv.ImageProcessorInterface
}

func NewRunDetectionUseCase(tf tensorflow.MachineLearningInterface, openCV opencv.ImageProcessorInterface) *RunDetectionUseCase {
	return &RunDetectionUseCase{
		Tf:     tf,
		OpenCV: openCV,
	}
}

// Runs all initial process to detect the objects in the received frame
func (rduc *RunDetectionUseCase) Execute(frame gocv.Mat) {
	bmpImageEncode, err := rduc.OpenCV.EncodeImageToBMP(frame)
	if err != nil {
		log.Fatalf("error encoding frame: %v", err)
	}
	defer bmpImageEncode.Close()

	// Make a tensor and an Image from the frame bytes
	tensor, _, err := rduc.Tf.MakeTensorFromImage(bmpImageEncode.GetBytes())
	if err != nil {
		log.Fatalf("error making input tensor: %v", err)
	}

	// Run inference on the newly made input tensor
	probabilities, classes, boxes, err := rduc.Tf.PredictObjectBoxes(tensor)
	if err != nil {
		log.Fatalf("error making prediction: %v", err)
	}

	// Only get the probabilities that have a percentage >= 0.5
	for probability := 0; probability < len(probabilities); probability++ {
		if probabilities[probability] > 0.5 {
			// Box coordinates come in as [y1, x1, y2, x2]
			boxCoordinateX1 := float64(boxes[probability][1])
			boxCoordinateX2 := float64(boxes[probability][3])
			boxCoordinateY1 := float64(boxes[probability][0])
			boxCoordinateY2 := float64(boxes[probability][2])

			// Convert box coordinates to pixel coordinates
			imageWidth := float64(frame.Cols())
			imageHeight := float64(frame.Rows())
			x1 := int(boxCoordinateX1 * imageWidth)
			y1 := int(boxCoordinateY1 * imageHeight)
			x2 := int(boxCoordinateX2 * imageWidth)
			y2 := int(boxCoordinateY2 * imageHeight)

			probabilityOutput := fmt.Sprintf(
				"OBJECT DETECTED: %s | COORDINATES: x1 %v - x2 %v - y1 %v - y2 %v",
				getNameOfDetectedObject(probability, probabilities, classes),
				x1,
				x2,
				y1,
				y2,
			)

			fmt.Println(probabilityOutput)

			// Draw a box around the objects
			rduc.OpenCV.DrawRectangle(frame, image.Rect(x1, y1, x2, y2))

			// Draw a text around the objects
			rduc.OpenCV.DrawText(
				frame,
				getNameOfDetectedObject(probability, probabilities, classes),
				image.Point{X: x1 + 10, Y: y1 - 20},
			)
		}
	}
}

// Returns a string with the name of the object that was detected along with the detection accuracy percentage | eg: person (90%)
func getNameOfDetectedObject(classId int, probabilities []float32, classes []float32) (label string) {
	index := int(classes[classId])
	label = fmt.Sprintf("%s (%2.0f%%)", tensorflow.COCO_SSD_LABELS[index], probabilities[classId]*100.0)

	return label
}
