package main

import (
	FLRubato "flhhe"
	"flhhe/configs"
	"flhhe/src/RtF"
	"flhhe/src/hhe_fedavg/keys_dealer"
	"flhhe/src/hhe_fedavg/server"
	"flhhe/src/utils"
	"fmt"
	"math"
	"path/filepath"
	"testing"
	"time"
)

func loadDecryptCompare(
	logger utils.Logger,
	rootPath string,
	weightIndex int,
	rubatoParams *keys_dealer.RubatoParams,
	hheComponents *keys_dealer.HHEComponents,
) {
	logger.PrintHeader(fmt.Sprintf("Testing avgFC%d", weightIndex))
	// Paths
	plainHEDecryptedAvgWeightsDir := filepath.Join(rootPath, configs.DecryptedWeights)
	logger.PrintFormatted("Plain HE Decrypted avg weights dir: %s", plainHEDecryptedAvgWeightsDir)
	avgCiphertextsDir := filepath.Join(rootPath, configs.HEEncryptedWeights, "avg")
	logger.PrintFormatted("Avg ciphertexts dir (from the HHE protocol): %s", avgCiphertextsDir)

	plainHEDecryptedAvgWeights := utils.LoadFromJSON(logger, plainHEDecryptedAvgWeightsDir, fmt.Sprintf("he_decrypted_avg_fc%d.json", weightIndex))
	logger.PrintFormatted("Plaintext avg weights type: %T and length: %d", plainHEDecryptedAvgWeights, len(plainHEDecryptedAvgWeights))

	logger.PrintFormatted("Rubato params num slots: %d", rubatoParams.Params.Slots())

	plainHEDecryptedAvgWeightsComplex := make([]complex128, rubatoParams.Params.Slots())
	for i := range len(plainHEDecryptedAvgWeights) {
		plainHEDecryptedAvgWeightsComplex[i] = complex(plainHEDecryptedAvgWeights[i], 0)
	}

	// Load and decrypt the heAvgWeights
	logger.PrintMessage("--- Decrypting the HE ciphertexts of the avg weights from the HHE protocol ---")
	heAvgWeights := server.LoadCipher(logger, weightIndex-1, avgCiphertextsDir, rubatoParams.Params)

	ckksEncoder := hheComponents.CkksEncoder
	ckksDecryptor := hheComponents.CkksDecryptor
	t := time.Now()
	decryptedAvgWeights := ckksEncoder.DecodeComplex(ckksDecryptor.DecryptNew(heAvgWeights), rubatoParams.Params.LogSlots())
	logger.PrintRunningTime("Time to decrypt the HE ciphertexts of the avg weights from the HHE protocol", t)
	logger.PrintFormatted("Decrypted avg weights type: %T and length: %d", decryptedAvgWeights, len(decryptedAvgWeights))

	// Calculate the error
	logger.PrintMessage("--- Calculating the error ---")
	logSlots := rubatoParams.Params.LogSlots()
	sigma := rubatoParams.Params.Sigma()

	logger.PrintFormatted("Level: %d (logQ = %d)", heAvgWeights.Level(), rubatoParams.Params.LogQLvl(heAvgWeights.Level()))
	logger.PrintFormatted("Scale: 2^%f", math.Log2(heAvgWeights.Scale()))
	logger.PrintFormatted("decryptedAvgWeights{%d}: [%6.10f %6.10f %6.10f %6.10f...]",
		len(decryptedAvgWeights), decryptedAvgWeights[0], decryptedAvgWeights[1], decryptedAvgWeights[2], decryptedAvgWeights[3])
	logger.PrintFormatted("plaintextAvgWeights{%d}: [%6.10f %6.10f %6.10f %6.10f...]",
		len(plainHEDecryptedAvgWeightsComplex), plainHEDecryptedAvgWeightsComplex[0], plainHEDecryptedAvgWeightsComplex[1], plainHEDecryptedAvgWeightsComplex[2], plainHEDecryptedAvgWeightsComplex[3])

	precisionStats := RtF.GetPrecisionStats(rubatoParams.Params, ckksEncoder, nil, plainHEDecryptedAvgWeightsComplex, decryptedAvgWeights, logSlots, sigma)
	fmt.Println(precisionStats.String())

	// Assert that precision values are in good range
	minRealThreshold := 18.0 // Log2
	minImagThreshold := 31.0 // Log2
	if real(precisionStats.MinPrecision) < minRealThreshold || imag(precisionStats.MinPrecision) < minImagThreshold {
		panic(fmt.Sprintf("Minimum precision below threshold: got (%.2f, %.2f), want at least (%.2f, %.2f)",
			real(precisionStats.MinPrecision), imag(precisionStats.MinPrecision), minRealThreshold, minImagThreshold))
	}

	avgRealThreshold := 21.0 // Log2
	avgImagThreshold := 36.0 // Log2
	if real(precisionStats.MeanPrecision) < avgRealThreshold || imag(precisionStats.MeanPrecision) < avgImagThreshold {
		panic(fmt.Sprintf("Average precision below threshold: got (%.2f, %.2f), want at least (%.2f, %.2f)",
			real(precisionStats.MeanPrecision), imag(precisionStats.MeanPrecision), avgRealThreshold, avgImagThreshold))
	}

	// Check error standard deviation (lower is better)
	errStdFThreshold := 24.0 // Log2
	errStdTThreshold := 16.0 // Log2
	if math.Log2(precisionStats.STDFreq) > errStdFThreshold {
		panic(fmt.Sprintf("Error stdF too high: got %.2f, want at most %.2f",
			math.Log2(precisionStats.STDFreq), errStdFThreshold))
	}
	if math.Log2(precisionStats.STDTime) > errStdTThreshold {
		panic(fmt.Sprintf("Error stdT too high: got %.2f, want at most %.2f",
			math.Log2(precisionStats.STDTime), errStdTThreshold))
	}

	utils.SaveComplexToJSON(logger, plainHEDecryptedAvgWeightsDir, fmt.Sprintf("hhe_decrypted_avg_fc%d.json", weightIndex), decryptedAvgWeights)
}

func TestHHEFedAvg(t *testing.T) {
	logger := utils.NewLogger(utils.DEBUG)
	rootPath := FLRubato.FindRootPath()

	paramIndex := RtF.RUBATO128L
	rubatoParams, hheComponents, _ := keys_dealer.RunKeysDealer(logger, rootPath, paramIndex)

	loadDecryptCompare(logger, rootPath, 1, rubatoParams, hheComponents) // test avgFC1
	loadDecryptCompare(logger, rootPath, 2, rubatoParams, hheComponents) // test avgFC2
}
