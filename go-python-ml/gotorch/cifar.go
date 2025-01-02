package main

import (
	"fmt"
	"image"
	_ "image/jpeg"
	"log"
	"os"

	"github.com/pytorch/go-torch"
	"github.com/pytorch/go-torch/torchvision"
)

func main() {
	// Load the trained model
	model, err := torch.LoadModel("cifar10_model.pt")
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	// Define the transformations for the input images
	transforms := torchvision.Transforms{
		torchvision.ToTensor(),
		torchvision.Normalize([]float32{0.5, 0.5, 0.5}, []float32{0.5, 0.5, 0.5}),
	}

	// Load and preprocess the input image
	imagePath := "input_image.jpg"
	imageFile, err := os.Open(imagePath)
	if err != nil {
		log.Fatal(err)
	}
	defer imageFile.Close()

	img, _, err := image.Decode(imageFile)
	if err != nil {
		log.Fatal(err)
	}

	input := transforms.Apply(img)

	// Perform inference with the model
	output := model.Forward(input)

	// Get the predicted class
	_, predictedClass := torch.Max(output, 1)

	// Map the class index to the class name
	classNames := []string{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"}
	className := classNames[predictedClass.Item()]

	fmt.Printf("Predicted class: %s\n", className)
}
