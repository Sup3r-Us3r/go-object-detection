package main

import (
	"fmt"
	"log"
	"path/filepath"
	"sync"

	"github.com/Sup3r-Us3r/go-object-detection/internal/domain/entity"
	"github.com/Sup3r-Us3r/go-object-detection/internal/infra/opencv"
	"github.com/Sup3r-Us3r/go-object-detection/internal/infra/tensorflow"
	"github.com/Sup3r-Us3r/go-object-detection/internal/usecase"
)

func worker(
	workerId int,
	camera entity.VideoStream,
	openCV *opencv.OpenCVImageProcessor,
	runDetectionUseCase *usecase.RunDetectionUseCase,
) {
	fmt.Printf("WORKER ID [%v] - DEVICE [%v]\n", workerId, camera.Label)

	openCV.CaptureStreamVideo(camera.Input, runDetectionUseCase.Execute)
}

// List of cameras to run object detection
func getCameras() (cameras []entity.VideoStream) {
	cameras = []entity.VideoStream{
		{
			ID:    "1",
			Label: "Camera 1",
			Input: "rtsp://host:port",
		},
	}

	return cameras
}

func main() {
	wg := sync.WaitGroup{}

	tf := tensorflow.NewTensorflowMachineLearning()
	openCV := opencv.NewOpenCVImageProcessor()

	modelPath, err := filepath.Abs("data/models/ssd_mobilenet_v1_coco_2018_01_28/saved_model")
	if err != nil {
		log.Fatal(err)
	}

	model, tfSession, err := tf.LoadSavedModel(modelPath)
	if err != nil {
		log.Fatal(err)
	}
	defer model.Session.Close()
	defer tfSession.Close()

	runDetectionUseCase := usecase.NewRunDetectionUseCase(tf, openCV)

	for index, camera := range getCameras() {
		wg.Add(1)
		defer wg.Done()

		go worker(index+1, camera, openCV, runDetectionUseCase)
	}

	wg.Wait()
}
