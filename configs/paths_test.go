package configs

import (
	FLRubato "flhhe"
	"fmt"
	"github.com/stretchr/testify/assert"
	"path/filepath"
	"testing"
)

func TestPaths(t *testing.T) {
	root := FLRubato.FindRootPath()
	fmt.Println(root)

	t.Run("Test configs directory paths", func(t *testing.T) {
		configPath := filepath.Join(root, Configs)
		fmt.Println("Expected: ", configPath)
		assert.DirExists(t, configPath)
	})

	t.Run("Test keys directory paths", func(t *testing.T) {
		keysPath := filepath.Join(root, Keys)
		fmt.Println("Expected: ", keysPath)
		assert.DirExists(t, keysPath)
	})

	t.Run("Test Mnist Weights directory paths", func(t *testing.T) {
		mnist := filepath.Join(root, MNIST)
		fmt.Println("Expected: ", mnist)
		assert.DirExists(t, mnist)
	})
}
