package utils

import (
	"encoding/json"
	"flhhe/configs"
	"os"
	"path/filepath"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

type ModelWeights struct {
	FC1          [][]float64 `json:"fc1"`
	FC2          [][]float64 `json:"fc2"`
	FC1Flatten   []float64
	FC2Flatten   []float64
	FC1Encrypted []*rlwe.Ciphertext
	FC2Encrypted []*rlwe.Ciphertext
}

func NewModelWeights() ModelWeights {
	return ModelWeights{}
}

// LoadWeights open the file path and load the model weights also will do the flattening
func (mw *ModelWeights) LoadWeights(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	if err := json.Unmarshal(data, &mw); err != nil {
		return err
	} else {
		mw.FC1Flatten = Flatten2D(mw.FC1)
		mw.FC2Flatten = Flatten2D(mw.FC2)
	}
	return nil
}

// Print2DLayerDimension print the dimensions of 2D model
func (mw *ModelWeights) Print2DLayerDimension(logger Logger) {
	logger.PrintFormatted("Shape: FC1[%d, %d]", len(mw.FC1), len(mw.FC1[0]))
	logger.PrintFormatted("Shape: FC2[%d, %d]", len(mw.FC2), len(mw.FC2[0]))
}

// Flatten2D converts a 2D slice into a 1D slice by concatenating all rows (row major packing)
func Flatten2D(matrix [][]float64) []float64 {
	totalLen := len(matrix) * len(matrix[0])
	flattened := make([]float64, 0, totalLen)
	for _, row := range matrix {
		flattened = append(flattened, row...)
	}
	return flattened
}

func OpenModelWeights(logger Logger, root string, weightFile string) ModelWeights {
	var err error
	weightDir := filepath.Join(root, configs.PlaintextWeights)

	weightPath := filepath.Join(weightDir, weightFile)
	logger.PrintFormatted("Loading weights from %s", weightPath)
	weights := NewModelWeights()
	err = weights.LoadWeights(weightPath)
	HandleError(err)

	return weights
}

// SaveToJSON saves a slice of float64 values to a JSON file
func SaveToJSON(logger Logger, weightDir string, filename string, data []float64) {
	// Create a file
	fileName := filepath.Join(weightDir, filename)
	file, err := os.Create(fileName)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	// Create an encoder and write the data
	encoder := json.NewEncoder(file)
	err = encoder.Encode(data)
	if err != nil {
		panic(err)
	}
	logger.PrintFormatted("Saved data to %s", fileName)
}

// SaveComplexToJSON saves a slice of complex128 values to a JSON file
func SaveComplexToJSON(logger Logger, weightDir string, filename string, data []complex128) {
	// Convert complex128 to float64 by taking the real part
	floatData := make([]float64, len(data))
	for i, val := range data {
		floatData[i] = real(val)
	}

	// Use existing saveToJSON function to save the float64 data
	SaveToJSON(logger, weightDir, filename, floatData)
}
