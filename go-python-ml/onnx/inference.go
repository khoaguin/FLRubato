package main

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	// This line _may_ be optional; by default the library will try to load
	// "onnxruntime.dll" on Windows, and "onnxruntime.so" on any other system.
	// For stability, it is probably a good idea to always set this explicitly.
	ort.SetSharedLibraryPath("/usr/lib/libonnxruntime.so")

	err := ort.InitializeEnvironment()
	if err != nil {
		panic(err)
	}
	defer ort.DestroyEnvironment()

	// For a slight performance boost and convenience when re-using existing
	// tensors, this library expects the user to create all input and output
	// tensors prior to creating the session. If this isn't ideal for your use
	// case, see the DynamicAdvancedSession type in the documnentation, which
	// allows input and output tensors to be specified when calling Run()
	// rather than when initializing a session.
	inputData := []float32{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
	inputShape := ort.NewShape(1, 10)
	inputTensor, _ := ort.NewTensor(inputShape, inputData)
	defer inputTensor.Destroy()

	outputShape := ort.NewShape(1, 5)
	outputTensor, _ := ort.NewEmptyTensor[float32](outputShape)
	defer outputTensor.Destroy()

	session, err := ort.NewAdvancedSession("model.onnx",
		[]string{"input"}, []string{"output"}, // Match the names from torch.onnx.export
		[]ort.Value{inputTensor}, []ort.Value{outputTensor}, nil)
	if err != nil {
		panic(err)
	}
	defer session.Destroy()

	// Get the weight tensor
	// The name "layer.weight" comes from your PyTorch model structure
	metadata, _ := session.GetModelMetadata()
	defer metadata.Destroy()
	des, _ := metadata.GetDescription()
	model_producer, _ := metadata.GetProducerName()
	model_version, _ := metadata.GetVersion()
	fmt.Printf("\nModel Description: %s\n", des)
	fmt.Printf("Model Producer: %s\n", model_producer)
	fmt.Printf("Model Version: %d\n", model_version)

	weightTensor := session.GetTensorByName("layer.weight")
	biasTensor := session.GetTensorByName("layer.bias")

	// Calling Run() will run the network, reading the current contents of the
	// input tensors and modifying the contents of the output tensors.
	err = session.Run()
	if err != nil {
		panic(err)
	}
	// Get a slice view of the output tensor's data.
	outputData := outputTensor.GetData()

	// print out the output data
	fmt.Println("Output:")
	for i := 0; i < len(outputData); i++ {
		print(outputData[i], " ")
	}

	// If you want to run the network on a different input, all you need to do
	// is modify the input tensor data (available via inputTensor.GetData())
	// and call Run() again.
	print("\n Done!!!!")
	// ...
}
