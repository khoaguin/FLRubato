package main

import (
	// "github.com/ldsec/lattigo/v2/ckks_fv"
	"fmt"

	"flhhe/rubato/ckks_fv"
)

func testRtFRubatoModDown(rubatoParam int, paramIndex int, radix int, fullCoeffs bool) {
	// var err error

	var hbtp *ckks_fv.HalfBootstrapper
	// var kgen ckks_fv.KeyGenerator
	// var fvEncoder ckks_fv.MFVEncoder
	// var ckksEncoder ckks_fv.CKKSEncoder
	// var ckksDecryptor ckks_fv.CKKSDecryptor
	// var sk *ckks_fv.SecretKey
	// var pk *ckks_fv.PublicKey
	// var fvEncryptor ckks_fv.MFVEncryptor
	// var fvEvaluator ckks_fv.MFVEvaluator
	// var plainCKKSRingTs []*ckks_fv.PlaintextRingT
	// var plaintexts []*ckks_fv.Plaintext
	// var rubato ckks_fv.MFVRubato

	// blocksize := ckks_fv.RubatoParams[rubatoParam].Blocksize
	// numRound := ckks_fv.RubatoParams[rubatoParam].NumRound
	// plainModulus := ckks_fv.RubatoParams[rubatoParam].PlainModulus
	// sigma := ckks_fv.RubatoParams[rubatoParam].Sigma

	// fmt.Printf("blocksize: [%d]\n", blocksize)

	fmt.Printf("hbtp: [%d]\n", hbtp)

}

func main() {
	testRtFRubatoModDown(4, 0, 2, false)
}
