package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
)

type Input struct {
	Operation string    `json:"operation"`
	Numbers   []float64 `json:"numbers"`
}

type Output struct {
	Result float64 `json:"result"`
}

func main() {
	// Read input from stdin
	inputBytes, err := io.ReadAll(os.Stdin)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
		os.Exit(1)
	}

	// Parse JSON input
	var input Input
	if err := json.Unmarshal(inputBytes, &input); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing JSON: %v\n", err)
		os.Exit(1)
	}

	// Perform calculation
	var result float64
	switch input.Operation {
	case "multiply":
		result = input.Numbers[0] * input.Numbers[1]
	case "add":
		result = input.Numbers[0] + input.Numbers[1]
	default:
		fmt.Fprintf(os.Stderr, "Unknown operation: %s\n", input.Operation)
		os.Exit(1)
	}

	// Return result as JSON
	output := Output{Result: result}
	outputBytes, err := json.Marshal(output)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating JSON output: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(string(outputBytes))
}
