package server

import (
	"flhhe/configs"
	"flhhe/src/RtF"
	"flhhe/src/hhe_fedavg/client"
	"flhhe/src/hhe_fedavg/keys_dealer"
	"flhhe/src/utils"
	"path/filepath"
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
	symKeyFVCiphertext := keys_dealer.LoadCiphertextArray(symCipherDir, rubatoParams.Params)

	for _, flClient := range flClients {
		var fvKeyStreams []*RtF.Ciphertext
		//fvKeyStreams = rubato.Crypt(nonces, counter, kCt, rubatoModDown)
		logger.PrintHeader("[Server - Offline] Evaluates the keystreams (Eval^{FV}) to produce V")
		t := time.Now()
		fvKeyStreams = rubato.CryptNoModSwitch(
			flClient.Nonces, 
			flClient.Counter, 
			symKeyFVCiphertext,
		) // Compute ciphertexts without modulus switching
		logger.PrintRunningTime("Time to evaluate the keystreams (Eval^{FV}) to produce V", t)
		logger.PrintFormatted("Keystreams dimension: [%d]", len(fvKeyStreams))
	}
}
