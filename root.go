package FLRubato

import (
	"path/filepath"
	"runtime"
)

func FindRootPath() string {
	_, filename, _, _ := runtime.Caller(0)
	projectRoot := filepath.Dir(filename) // Adjust as necessary
	return projectRoot
}
