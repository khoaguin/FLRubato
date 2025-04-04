package main

import (
	"time"

	FLRubato "flhhe"
	"flhhe/src/RtF"
	"flhhe/src/utils"

	"flhhe/src/hhe_fedavg/client"
	"flhhe/src/hhe_fedavg/keys_dealer"
	"flhhe/src/hhe_fedavg/server"
)

//================= How Rubato Works: =================//
/*
	1. Generating the HE context keys
	2. Generating HalfBoot Keys
	3. Data Generation (or Collection)
		- Rubato, depending on the parameter selection, has a fixed block size as BS={16, 36, 64}
		|-> with corresponding plaintext modulus logP: {26, 25, 25} and LogN = 16, logSlots = 15,
		|-> with a scaling factor logSF = {45}. Therefore, we can make a matrix as following
		|-> data = [BS-4][N]float64.
	4. Generation of the SymKey, Nonce, and Counter
		- The SymKey will be as K = [BS]uint64
		- The Nonce will be as nonces = [N][8]byte
		- The Counter will be as counter = [8]byte
	5. Symmetric keystream generation
		- Using plainRubato(BS, numRound, nonces[i], counter, symkey, pMod, sigma) we can generate
		|-> a keystream as ks = [N][BS-4]uint64, (Each element of keystream has 4 columns less than
		|-> the actual block size, since we drop 4 elements because of the noise addition)
	6. Symmetric Data Encryption
		- To encrypt the data using the symmetric keystream, we first move data to the plaintext's coefficients
		|-> and then use CKKS Encode function to have data as a CKKS polynomial ring.
		|-> Finally, one can add the keystream to the polynomial coefficients and do the modular reduction.
	7. Scaling up the data
		- In this step, we Scale Up the encrypted data, PlaintextRingT (R_p) into a Plaintext (R_q),
		|-> so we get a plaintext in RingQ with level=0
	8. Encrypting the SymKey
		- In this step, we will encrypt the SymKey homomorphically using MFVRubato as kCT
	9. Offline computation
		- This step includes the generation of homomorphic keystream using kCT and then switching
		|-> the modulus down to stcModDown, so in the end we will have the homomorphic keystream
		|-> ciphertext with the same modulus as encrypted data
	10. Online computation
		- This step includes the homomorphic decryption of the symmetrically encrypted data (aka
		|-> transciphering) to have homomorphically encrypted data (CKKS ciphertext) using HalfBoot
		|-> function. One can then use the result ciphertext to do the rest of the HE application
		|->	evaluations. Keep in mind that one can also store this ciphertext to avoid the repeat of
		|-> the heavy computation for transciphering.
*/
//================= How Rubato Works End =================//
func main() {
	logger := utils.NewLogger(utils.DEBUG)
	rootPath := FLRubato.FindRootPath()
	paramIndex := RtF.RUBATO128L

	t := time.Now()

	rubatoParams, hheComponents, rubato := keys_dealer.RunKeysDealer(logger, rootPath, paramIndex)
	logger.PrintFormatted("Rubato Parameters: %+v", rubatoParams)
	logger.PrintFormatted("HHE Components: %+v", hheComponents)
	logger.PrintFormatted("Rubato Instance Addr: %+v", &rubato)

	flClients := make([]*client.FLClient, 3)
	flClients[0] = client.RunFLClient(logger, rootPath, rubatoParams, hheComponents, "mnist_weights_exclude_137.json", "client2")
	flClients[1] = client.RunFLClient(logger, rootPath, rubatoParams, hheComponents, "mnist_weights_exclude_258.json", "client2")
	flClients[2] = client.RunFLClient(logger, rootPath, rubatoParams, hheComponents, "mnist_weights_exclude_469.json", "client3")
	server.RunFLServer(logger, rootPath, flClients, rubatoParams, hheComponents, rubato)

	logger.PrintRunningTime("Total time to run the program", t)
}
