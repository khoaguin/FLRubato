package server

import (
	"flhhe/configs"
	"flhhe/src/RtF"
	"flhhe/src/hhe_fedavg/client"
	"flhhe/src/hhe_fedavg/keys_dealer"
	"flhhe/src/utils"
	"fmt"
	"math"
	"path/filepath"
	"strconv"
	"time"
)

func RunFLServer(
	logger utils.Logger,
	rootPath string,
	flClients []*client.FLClient,
	rubatoParams *keys_dealer.RubatoParams,
	hheComponents *keys_dealer.HHEComponents,
	rubato RtF.MFVRubato,
) {
	logger.PrintHeader("--- Aggregator Server ---")
	logger.PrintHeader("[Server - Offline] Loading the FV encrypted symmetric key")
	keysDir := filepath.Join(rootPath, configs.Keys)
	symCipherDir := filepath.Join(keysDir, configs.SymmetricKeyCipherDir)
	logger.PrintFormatted("Symmetric key ciphertext directory: %s", symCipherDir)
	symKeyFVCiphertext := keys_dealer.LoadCiphertextArray(symCipherDir, rubatoParams.Params)

	for _, flClient := range flClients {
		// Reset the rubato instance before processing each client
		rubato.Reset(rubatoParams.RubatoModDown[0])

		logger.PrintHeader("[Server - Offline] Evaluates the keystreams (Eval^{FV}) to produce V")
		t := time.Now()
		fvKeyStreams := rubato.CryptNoModSwitch(
			flClient.Nonces,
			flClient.Counter,
			symKeyFVCiphertext,
		)
		logger.PrintRunningTime("Time to evaluate the keystreams (Eval^{FV}) to produce V", t)
		logger.PrintFormatted("Keystreams dimension: [%d]", len(fvKeyStreams))

		logger.PrintHeader("[Server - Offline] Performs linear transformation SlotToCoeffs^{FV} to produce Z")
		t = time.Now()
		for i := range rubatoParams.OutputSize {
			fvKeyStreams[i] = hheComponents.FvEvaluator.SlotsToCoeffs(fvKeyStreams[i], rubatoParams.StcModDown)
			hheComponents.FvEvaluator.ModSwitchMany(fvKeyStreams[i], fvKeyStreams[i], fvKeyStreams[i].Level())
		}
		logger.PrintRunningTime("Time to perform linear transformation SlotToCoeffs^{FV} to produce Z", t)

		logger.PrintHeader("[Server - Online] Scale up the symmetric ciphertext (Scale{FV}) into FV-ciphretext space (produce C)")
		plainCKKSRingTs := flClient.SymmCipher
		t = time.Now()
		plaintexts := make([]*RtF.Plaintext, rubatoParams.OutputSize)
		for s := range rubatoParams.OutputSize {
			plaintexts[s] = RtF.NewPlaintextFVLvl(rubatoParams.Params, 0)
			hheComponents.FvEncoder.FVScaleUp(plainCKKSRingTs[s], plaintexts[s])
		}
		logger.PrintRunningTime("Time to scale up the symmetric ciphertext into FV-ciphertext space", t)

		var ctBoot *RtF.Ciphertext
		logger.PrintHeader("[Server - Online] Transciphering the symmetric ciphertext into CKKS ciphertext (produce M)")
		for s := range rubatoParams.OutputSize {
			// Encrypt and mod switch to the lowest level
			ciphertext := RtF.NewCiphertextFVLvl(rubatoParams.Params, 1, 0)
			ciphertext.Value()[0] = plaintexts[s].Value()[0].CopyNew()
			logger.PrintMessage("Subtracting the homomorphically evaluated keystream Z from the symmetric ciphertext C (produce X)")
			hheComponents.FvEvaluator.Sub(ciphertext, fvKeyStreams[s], ciphertext)
			hheComponents.FvEvaluator.TransformToNTT(ciphertext, ciphertext)
			ciphertext.SetScale(math.Exp2(math.Round(math.Log2(float64(rubatoParams.Params.Qi()[0]) / float64(rubatoParams.Params.PlainModulus()) * rubatoParams.MessageScaling))))
			// Half-Bootstrap the ciphertext (homomorphic evaluation of ModRaise -> SubSum -> CtS -> EvalMod)
			// It takes a ciphertext at level 0 (if not at level 0, then it will reduce it to level 0)
			// and returns a ciphertext at level MaxLevel - k, where k is the depth of the bootstrapping circuit.
			// The difference from the bootstrapping is that the last StC is missing.
			// CAUTION: the scale of the ciphertext MUST be equal (or very close) to params.Scale
			// To equalize the scale, the function evaluator.SetScale(ciphertext, parameters.Scale) can be used at the expense of one level.
			logger.PrintMessage("Halfboot X and outputs a CKKS-ciphertext M containing the CKKS encrypted messages in its slots")
			t = time.Now()
			ctBoot, _ = hheComponents.HalfBootstrapper.HalfBoot(ciphertext, false)
			logger.PrintRunningTime("HalfBoot", t)

			valuesWant := make([]complex128, rubatoParams.Params.Slots())
			for i := range rubatoParams.Params.Slots() {
				valuesWant[i] = complex(flClient.PlaintextData[s][i], 0)
			}

			printString := fmt.Sprintf("Precision of HalfBoot(ciphertext[%d])", s)
			logger.PrintHeader(printString)
			printDebug(
				logger,
				rubatoParams.Params,
				ctBoot,
				valuesWant,
				hheComponents.CkksDecryptor,
				hheComponents.CkksEncoder,
			)

			cipherDir := filepath.Join(rootPath, configs.Ciphertexts, flClient.ClientID)
			// save the CKKS ciphertext in a file for further computation
			SaveCipher(logger, s, cipherDir, ctBoot)
		}
	}
}

func SaveCipher(
	logger utils.Logger,
	index int,
	ciphersDir string,
	ciphertext *RtF.Ciphertext) {
	logger.PrintHeader("Save the CKKS ciphertext in a file for further computation")
	var err error
	fileName := configs.CtNameFix + strconv.Itoa(index) + configs.CtFormat
	err = utils.Serialize(ciphertext, filepath.Join(ciphersDir, fileName))
	utils.HandleError(err)
	logger.PrintFormatted("Ciphertext saved to %s", filepath.Join(ciphersDir, fileName))
}

// LoadCipher save a ciphertext in the provided path
func LoadCipher(
	logger utils.Logger,
	ciphersDir string,
	fileName string,
	params *RtF.Parameters) any {
	logger.PrintHeader("Load the CKKS ciphertext from the file")
	ciphertext := RtF.NewCiphertextFVLvl(params, 1, 0)
	err := utils.Deserialize(ciphertext, filepath.Join(ciphersDir, fileName))
	utils.HandleError(err)
	return ciphertext
}

func printDebug(
	logger utils.Logger,
	params *RtF.Parameters,
	ciphertext *RtF.Ciphertext,
	valuesWant []complex128,
	decryptor RtF.CKKSDecryptor,
	encoder RtF.CKKSEncoder) {
	if utils.DEBUG {

		valuesTest := encoder.DecodeComplex(decryptor.DecryptNew(ciphertext), params.LogSlots())
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
