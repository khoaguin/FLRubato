package server

import (
	"flhhe/configs"
	"flhhe/src/RtF"
	"flhhe/src/hhe_fedavg/client"
	"flhhe/src/hhe_fedavg/keys_dealer"
	"flhhe/src/utils"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

// RunFLServer is the main entry point for the Federated Learning server
func RunFLServer(
	logger utils.Logger,
	rootPath string,
	flClients []*client.FLClient,
	rubatoParams *keys_dealer.RubatoParams,
	hheComponents *keys_dealer.HHEComponents,
	rubato RtF.MFVRubato,
) {
	logger.PrintHeader("--- Aggregator Server ---")

	// Load the FV encrypted symmetric key
	symKeyFVCiphertext := loadSymmetricKey(logger, rootPath, rubatoParams)

	// Process each client
	for _, flClient := range flClients {
		processClient(
			logger,
			rootPath,
			flClient,
			rubatoParams,
			hheComponents,
			rubato,
			symKeyFVCiphertext,
		)
	}

	// Load the ciphertexts and do HEFedAvg
	heFedAvg(logger, rootPath, flClients, rubatoParams, hheComponents)

}

// loadSymmetricKey loads the FV encrypted symmetric key
func loadSymmetricKey(
	logger utils.Logger,
	rootPath string,
	rubatoParams *keys_dealer.RubatoParams,
) []*RtF.Ciphertext {
	logger.PrintHeader("[Server - Offline] Loading the FV encrypted symmetric key")
	keysDir := filepath.Join(rootPath, configs.Keys)
	symCipherDir := filepath.Join(keysDir, configs.SymmetricKeyCipherDir)
	logger.PrintFormatted("Symmetric key ciphertext directory: %s", symCipherDir)
	return keys_dealer.LoadCiphertextArray(symCipherDir, rubatoParams.Params)
}

// processClient handles the processing of a single client
func processClient(
	logger utils.Logger,
	rootPath string,
	flClient *client.FLClient,
	rubatoParams *keys_dealer.RubatoParams,
	hheComponents *keys_dealer.HHEComponents,
	rubato RtF.MFVRubato,
	symKeyFVCiphertext []*RtF.Ciphertext,
) {
	// Reset the rubato instance before processing
	rubato.Reset(rubatoParams.RubatoModDown[0])

	// Generate and process keystreams (Z)
	fvKeyStreams := generateKeystreams(logger, flClient, rubatoParams, hheComponents, rubato, symKeyFVCiphertext)

	// Create plaintexts from symmetric ciphertexts (C)
	plaintexts := fvScaleUpSymCipher(logger, flClient, rubatoParams, hheComponents)

	// Process each output size and generate CKKS ciphertexts
	processCiphertexts(logger, rootPath, flClient, rubatoParams, hheComponents, fvKeyStreams, plaintexts)
}

// generateKeystreams evaluates the keystreams and performs linear transformation
func generateKeystreams(
	logger utils.Logger,
	flClient *client.FLClient,
	rubatoParams *keys_dealer.RubatoParams,
	hheComponents *keys_dealer.HHEComponents,
	rubato RtF.MFVRubato,
	symKeyFVCiphertext []*RtF.Ciphertext,
) []*RtF.Ciphertext {
	// Evaluate keystreams
	logger.PrintHeader("[Server - Offline] Evaluates the keystreams (Eval^{FV}) to produce V")
	t := time.Now()
	fvKeyStreams := rubato.CryptNoModSwitch(
		flClient.Nonces,
		flClient.Counter,
		symKeyFVCiphertext,
	)
	logger.PrintRunningTime("Time to evaluate the keystreams (Eval^{FV}) to produce V", t)
	logger.PrintFormatted("Keystreams dimension: [%d]", len(fvKeyStreams))

	// Perform linear transformation
	logger.PrintHeader("[Server - Offline] Performs linear transformation SlotToCoeffs^{FV} to produce Z")
	t = time.Now()
	for i := range rubatoParams.OutputSize {
		fvKeyStreams[i] = hheComponents.FvEvaluator.SlotsToCoeffs(fvKeyStreams[i], rubatoParams.StcModDown)
		hheComponents.FvEvaluator.ModSwitchMany(fvKeyStreams[i], fvKeyStreams[i], fvKeyStreams[i].Level())
	}
	logger.PrintRunningTime("Time to perform linear transformation SlotToCoeffs^{FV} to produce Z", t)

	return fvKeyStreams
}

// fvScaleUpSymCipher scales up the symmetric ciphertext into FV-ciphertext space
func fvScaleUpSymCipher(
	logger utils.Logger,
	flClient *client.FLClient,
	rubatoParams *keys_dealer.RubatoParams,
	hheComponents *keys_dealer.HHEComponents,
) []*RtF.Plaintext {
	logger.PrintHeader("[Server - Online] Scale up the symmetric ciphertext (Scale{FV}) into FV-ciphretext space (produce C)")
	plainCKKSRingTs := flClient.SymmCipher
	t := time.Now()
	plaintexts := make([]*RtF.Plaintext, rubatoParams.OutputSize)
	for s := range rubatoParams.OutputSize {
		plaintexts[s] = RtF.NewPlaintextFVLvl(rubatoParams.Params, 0)
		hheComponents.FvEncoder.FVScaleUp(plainCKKSRingTs[s], plaintexts[s])
	}
	logger.PrintRunningTime("Time to scale up the symmetric ciphertext into FV-ciphertext space", t)

	return plaintexts
}

// processCiphertexts handles the transciphering of symmetric ciphertext into CKKS ciphertext (M)
func processCiphertexts(
	logger utils.Logger,
	rootPath string,
	flClient *client.FLClient,
	rubatoParams *keys_dealer.RubatoParams,
	hheComponents *keys_dealer.HHEComponents,
	fvKeyStreams []*RtF.Ciphertext,
	plaintexts []*RtF.Plaintext,
) {
	logger.PrintHeader("[Server - Online] Transciphering the symmetric ciphertext into CKKS ciphertext (produce M)")

	for s := range rubatoParams.OutputSize {
		ciphertext := createInitialCiphertext(rubatoParams, plaintexts, s)

		logger.PrintMessage("Subtracting the homomorphically evaluated keystream Z from the symmetric ciphertext C (produce X)")
		hheComponents.FvEvaluator.Sub(ciphertext, fvKeyStreams[s], ciphertext)
		hheComponents.FvEvaluator.TransformToNTT(ciphertext, ciphertext)
		setScale(ciphertext, rubatoParams)

		// Perform half-bootstrapping
		ctBoot := performHalfBoot(logger, ciphertext, hheComponents)

		// Generate debug values
		valuesWant := generateDebugValues(flClient, rubatoParams, s)

		// Print debug information
		printString := fmt.Sprintf("Precision of HalfBoot(ciphertext[%d])", s)
		logger.PrintHeader(printString)
		PrintDebug(logger, rubatoParams.Params, ctBoot, valuesWant, hheComponents.CkksDecryptor, hheComponents.CkksEncoder)

		// Save the ciphertext
		cipherDir := filepath.Join(rootPath, configs.Ciphertexts, flClient.ClientID)
		SaveCipher(logger, s, cipherDir, ctBoot)
	}
}

// createInitialCiphertext creates and initializes a new ciphertext
func createInitialCiphertext(
	rubatoParams *keys_dealer.RubatoParams,
	plaintexts []*RtF.Plaintext,
	index int,
) *RtF.Ciphertext {
	ciphertext := RtF.NewCiphertextFVLvl(rubatoParams.Params, 1, 0)
	ciphertext.Value()[0] = plaintexts[index].Value()[0].CopyNew()
	return ciphertext
}

// setScale sets the appropriate scale for the ciphertext
func setScale(
	ciphertext *RtF.Ciphertext,
	rubatoParams *keys_dealer.RubatoParams,
) {
	scale := math.Exp2(math.Round(math.Log2(float64(rubatoParams.Params.Qi()[0]) /
		float64(rubatoParams.Params.PlainModulus()) * rubatoParams.MessageScaling)))
	ciphertext.SetScale(scale)
}

// performHalfBoot performs the half-bootstrapping operation (M)
func performHalfBoot(
	logger utils.Logger,
	ciphertext *RtF.Ciphertext,
	hheComponents *keys_dealer.HHEComponents,
) *RtF.Ciphertext {
	logger.PrintMessage("Halfboot X and outputs a CKKS-ciphertext M containing the CKKS encrypted messages in its slots")
	t := time.Now()
	ctBoot, _ := hheComponents.HalfBootstrapper.HalfBoot(ciphertext, false)
	logger.PrintRunningTime("HalfBoot", t)
	return ctBoot
}

func heFedAvg(
	logger utils.Logger,
	rootPath string,
	flClients []*client.FLClient,
	rubatoParams *keys_dealer.RubatoParams,
	hheComponents *keys_dealer.HHEComponents,
) {
	logger.PrintHeader("[Server - Online] HEFedAvg")

	// Load the ciphertexts
	ciphertexts := make([][]*RtF.Ciphertext, len(flClients))
	for i := range flClients {
		ciphertexts[i] = make([]*RtF.Ciphertext, rubatoParams.OutputSize)
		cipherDir := filepath.Join(rootPath, configs.Ciphertexts, flClients[i].ClientID)
		for j := range rubatoParams.OutputSize {
			ciphertexts[i][j] = LoadCipher(logger, j, cipherDir, rubatoParams.Params)
		}
	}
	logger.PrintFormatted("Ciphertexts: %+v", ciphertexts)

	// Do HEFedAvg
	avgCiphertexts := make([]*RtF.Ciphertext, rubatoParams.OutputSize)
	for i := range rubatoParams.OutputSize {
		avgCiphertexts[i] = ciphertexts[0][i].CopyNew().Ciphertext()
		for j := 1; j < len(flClients); j++ {
			avgCiphertexts[i] = hheComponents.CkksEvaluator.AddNew(avgCiphertexts[i], ciphertexts[j][i])
		}
	}
	for i := range rubatoParams.OutputSize {
		avgCiphertexts[i] = hheComponents.CkksEvaluator.MultByConstNew(avgCiphertexts[i], 1/float64(len(flClients)))
	}
	logger.PrintFormatted("AvgCiphertexts: %+v", avgCiphertexts)

	// Save the average ciphertexts
	avgCiphertextsDir := filepath.Join(rootPath, configs.Ciphertexts, "avg")
	os.MkdirAll(avgCiphertextsDir, 0755)
	for i := range rubatoParams.OutputSize {
		SaveCipher(logger, i, avgCiphertextsDir, avgCiphertexts[i])
	}
	logger.PrintFormatted("AvgCiphertexts saved to %s", avgCiphertextsDir)

	logger.PrintHeader("[Server - Online] HEFedAvg done")
}

// generateDebugValues creates values for debugging and precision checking
func generateDebugValues(
	flClient *client.FLClient,
	rubatoParams *keys_dealer.RubatoParams,
	index int,
) []complex128 {
	valuesWant := make([]complex128, rubatoParams.Params.Slots())
	for i := range rubatoParams.Params.Slots() {
		valuesWant[i] = complex(flClient.PlaintextData[index][i], 0)
	}
	return valuesWant
}

func SaveCipher(
	logger utils.Logger,
	index int,
	ciphersDir string,
	ciphertext *RtF.Ciphertext) {
	var err error
	fileName := configs.CtNameFix + strconv.Itoa(index) + configs.CtFormat
	err = utils.Serialize(ciphertext, filepath.Join(ciphersDir, fileName))
	utils.HandleError(err)
	logger.PrintFormatted("Ciphertext saved to %s", filepath.Join(ciphersDir, fileName))
}

// LoadCipher loads a ciphertext from the provided path
func LoadCipher(
	logger utils.Logger,
	index int,
	ciphersDir string,
	params *RtF.Parameters) *RtF.Ciphertext {
	fileName := configs.CtNameFix + strconv.Itoa(index) + configs.CtFormat
	ciphertext := RtF.NewCiphertextFVLvl(params, 1, 0)
	err := utils.Deserialize(ciphertext, filepath.Join(ciphersDir, fileName))
	utils.HandleError(err)
	return ciphertext
}

func PrintDebug(
	logger utils.Logger,
	params *RtF.Parameters,
	ciphertext *RtF.Ciphertext,
	valuesWant []complex128,
	decryptor RtF.CKKSDecryptor,
	encoder RtF.CKKSEncoder) {
	if utils.DEBUG {
		logger.PrintHeader("--- Print Debug ---")
		valuesTest := encoder.DecodeComplex(decryptor.DecryptNew(ciphertext), params.LogSlots())
		logger.PrintFormatted("ValuesTest length: %d", len(valuesTest))

		logSlots := params.LogSlots()
		sigma := params.Sigma()

		logger.PrintFormatted("Level: %d (logQ = %d)", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
		logger.PrintFormatted("Scale: 2^%f", math.Log2(ciphertext.Scale()))
		logger.PrintFormatted("ValuesTest{%d}: [%6.10f %6.10f %6.10f %6.10f...]", len(valuesTest), valuesTest[0], valuesTest[1], valuesTest[2], valuesTest[3])
		logger.PrintFormatted("ValuesWant{%d}: [%6.10f %6.10f %6.10f %6.10f...]", len(valuesWant), valuesWant[0], valuesWant[1], valuesWant[2], valuesWant[3])

		precisionStats := RtF.GetPrecisionStats(params, encoder, nil, valuesWant, valuesTest, logSlots, sigma)
		fmt.Println(precisionStats.String())
	}
}
