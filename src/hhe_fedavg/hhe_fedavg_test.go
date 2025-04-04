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
)

func TestHHEFedAvg(t *testing.T) {
	logger := utils.NewLogger(utils.DEBUG)
	rootPath := FLRubato.FindRootPath()

	// Paths
	plaintextAvgWeightsDir := filepath.Join(rootPath, configs.MNIST)
	logger.PrintFormatted("Plaintext avg weights dir: %s", plaintextAvgWeightsDir)
	avgCiphertextsDir := filepath.Join(rootPath, configs.Ciphertexts, "avg")
	logger.PrintFormatted("Avg ciphertexts dir: %s", avgCiphertextsDir)

	plaintextAvgFC2 := utils.LoadFromJSON(logger, plaintextAvgWeightsDir, "plaintext_avg_fc2.json")
	logger.PrintFormatted("Plaintext avg weights type: %T", plaintextAvgFC2)

	paramIndex := RtF.RUBATO128L
	rubatoParams, hheComponents, _ := keys_dealer.RunKeysDealer(logger, rootPath, paramIndex)

	logger.PrintHeader("Generating the plaintext values in complex128")
	plaintextAvgFC2Complex := make([]complex128, rubatoParams.Params.Slots())
	for i := range len(plaintextAvgFC2) {
		plaintextAvgFC2Complex[i] = complex(plaintextAvgFC2[i], 0)
	}

	// Load the average ciphertexts
	logger.PrintHeader("Loading the average ciphertexts")
	avgCiphertexts := make([]*RtF.Ciphertext, rubatoParams.OutputSize)
	for i := range rubatoParams.OutputSize {
		avgCiphertexts[i] = server.LoadCipher(logger, i, avgCiphertextsDir, rubatoParams.Params)
	}

	// Decrypt the ciphertexts
	logger.PrintHeader("Decrypting the ciphertexts")
	ckksEncoder := hheComponents.CkksEncoder
	ckksDecryptor := hheComponents.CkksDecryptor
	decryptedAvgWeights := ckksEncoder.DecodeComplex(ckksDecryptor.DecryptNew(avgCiphertexts[2]), rubatoParams.Params.LogSlots())
	logger.PrintFormatted("Decrypted avg weights length: %d", len(decryptedAvgWeights))

	// Calculate the error
	logger.PrintHeader("Calculating the error")
	logSlots := rubatoParams.Params.LogSlots()
	sigma := rubatoParams.Params.Sigma()

	logger.PrintFormatted("Level: %d (logQ = %d)", avgCiphertexts[2].Level(), rubatoParams.Params.LogQLvl(avgCiphertexts[2].Level()))
	logger.PrintFormatted("Scale: 2^%f", math.Log2(avgCiphertexts[2].Scale()))
	logger.PrintFormatted("decryptedAvgWeights{%d}: [%6.10f %6.10f %6.10f %6.10f...]",
		len(decryptedAvgWeights), decryptedAvgWeights[0], decryptedAvgWeights[1], decryptedAvgWeights[2], decryptedAvgWeights[3])
	logger.PrintFormatted("plaintextAvgWeights{%d}: [%6.10f %6.10f %6.10f %6.10f...]",
		len(plaintextAvgFC2Complex), plaintextAvgFC2Complex[0], plaintextAvgFC2Complex[1], plaintextAvgFC2Complex[2], plaintextAvgFC2Complex[3])

	precisionStats := RtF.GetPrecisionStats(rubatoParams.Params, ckksEncoder, nil, plaintextAvgFC2Complex, decryptedAvgWeights, logSlots, sigma)
	fmt.Println(precisionStats.String())

	// Assert that precision values are in good range
	minRealThreshold := 19.0 // Log2
	minImagThreshold := 31.0 // Log2
	if real(precisionStats.MinPrecision) < minRealThreshold || imag(precisionStats.MinPrecision) < minImagThreshold {
		t.Errorf("Minimum precision below threshold: got (%.2f, %.2f), want at least (%.2f, %.2f)",
			real(precisionStats.MinPrecision), imag(precisionStats.MinPrecision), minRealThreshold, minImagThreshold)
	}

	avgRealThreshold := 21.0 // Log2
	avgImagThreshold := 36.0 // Log2
	if real(precisionStats.MeanPrecision) < avgRealThreshold || imag(precisionStats.MeanPrecision) < avgImagThreshold {
		t.Errorf("Average precision below threshold: got (%.2f, %.2f), want at least (%.2f, %.2f)",
			real(precisionStats.MeanPrecision), imag(precisionStats.MeanPrecision), avgRealThreshold, avgImagThreshold)
	}

	// Check error standard deviation (lower is better)
	errStdFThreshold := 24.0 // Log2
	errStdTThreshold := 16.0 // Log2
	if math.Log2(precisionStats.STDFreq) > errStdFThreshold {
		t.Errorf("Error stdF too high: got %.2f, want at most %.2f",
			math.Log2(precisionStats.STDFreq), errStdFThreshold)
	}
	if math.Log2(precisionStats.STDTime) > errStdTThreshold {
		t.Errorf("Error stdT too high: got %.2f, want at most %.2f",
			math.Log2(precisionStats.STDTime), errStdTThreshold)
	}

}
