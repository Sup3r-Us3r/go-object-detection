package opencv

import (
	"fmt"
	"image"
	"image/color"

	"gocv.io/x/gocv"
	_ "golang.org/x/image/bmp"
)

type ImageProcessorInterface interface {
	CaptureStreamVideo(input interface{}, runDetectionCallback func(currentFrame gocv.Mat)) error
	EncodeImageToBMP(currentFrame gocv.Mat) (output *gocv.NativeByteBuffer, err error)
	DrawRectangle(currentImage gocv.Mat, coordinates image.Rectangle)
	DrawText(currentImage gocv.Mat, text string, coordinates image.Point)
}

type OpenCVImageProcessor struct{}

func NewOpenCVImageProcessor() *OpenCVImageProcessor {
	return &OpenCVImageProcessor{}
}

// Start capturing video frames by Device ID, video file, RTSP/URL, or GStreamer pipeline
func (oip *OpenCVImageProcessor) CaptureStreamVideo(input interface{}, runDetectionCallback func(currentFrame gocv.Mat)) error {
	videoStream, err := gocv.OpenVideoCapture(input)
	if err != nil {
		return fmt.Errorf("unable to open video capture: %v", err)
	}

	frame := gocv.NewMat()

	window := gocv.NewWindow("Golang Object Detection")

	if ok := videoStream.Read(&frame); !ok {
		return fmt.Errorf("unable to capture image from camera")
	}

	for {
		videoStream.Read(&frame)

		if frame.Empty() {
			continue
		}

		runDetectionCallback(frame)

		window.IMShow(frame)
		window.WaitKey(1)
	}
}

// Compresses the image and stores it in the returned memory buffer, using .bmp extension
func (oip *OpenCVImageProcessor) EncodeImageToBMP(currentFrame gocv.Mat) (output *gocv.NativeByteBuffer, err error) {
	buf, err := gocv.IMEncode(".bmp", currentFrame)
	if err != nil {
		return nil, err
	}

	return buf, nil
}

// Draws a rectangle on the processed frame at the passed coordinates
func (oip *OpenCVImageProcessor) DrawRectangle(currentImage gocv.Mat, coordinates image.Rectangle) {
	gocv.Rectangle(
		&currentImage,                           // image
		coordinates,                             // rectangle
		color.RGBA{R: 64, G: 255, B: 134, A: 1}, // rgba color
		2,                                       // thickness
	)
}

// Draws a text on the processed frame at the passed coordinates
func (oip *OpenCVImageProcessor) DrawText(currentImage gocv.Mat, text string, coordinates image.Point) {
	gocv.PutText(
		&currentImage,                           //image
		text,                                    // text
		coordinates,                             // image point
		gocv.FontHersheySimplex,                 // font face
		0.6,                                     // font scale
		color.RGBA{R: 64, G: 255, B: 134, A: 1}, // color
		2,                                       // thickness
	)
}
