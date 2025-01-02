// package main

// import (
// 	"fmt"
// 	"log"
// 	"math/rand"

// 	"github.com/orktes/go-torch"
// )

// func main() {
// 	// Load the model
// 	// Load model
// 	model, _ := torch.LoadJITModule("simple_model.pt")

// 	// Create input tensor
// 	// Create sample input (10 features)
// 	data := make([]float32, 10)
// 	for i := range data {
// 		data[i] = float32(rand.NormFloat64())
// 	}

// 	// Run inference
// 	output, err := model.Forward(data)
// 	if err != nil {
// 		log.Fatalf("Failed to run inference: %v", err)
// 	}

// 	// Get the result
// 	fmt.Printf("Model Output:\n%v\n", output.(*torch.Tensor).Value())

// 	// For more detailed output you could also print:
// 	fmt.Printf("Output Shape: %v\n", output.(*torch.Tensor).Shape())
// 	fmt.Printf("Output Type: %v\n", output.(*torch.Tensor).Dtype())
// }
