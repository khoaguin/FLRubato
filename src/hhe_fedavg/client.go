package main

import "flhhe/src/utils"

func RunFLClient(
	logger utils.Logger,
	rootPath string,
	weightPath string,
	// params *RtF.Parameters,
	// fullCoffs bool,
) {
	logger.PrintHeader("FLRubato Client 01")
	logger.PrintHeader("[Client - Initialization]: Load plaintext weights from JSON (after training in python)")
	modelWeights := utils.OpenModelWeights(logger, rootPath, weightPath)
	modelWeights.Print2DLayerDimension(logger)

	logger.PrintHeader("[Client] Preparing the data")
	// data := preparingData(logger, 3, params, modelWeights)

}
