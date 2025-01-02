// Source: https://github.com/tuneinsight/lattigo/blob/v6.1.0/examples/singleparty/tutorials/ckks/main.go#L209

package main

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"golang.org/x/exp/rand"
)

func main() {
	// =================================
	// Instantiating the ckks.Parameters
	// =================================
	var err error
	var params ckks.Parameters
	if params, err = ckks.NewParametersFromLiteral(
		ckks.ParametersLiteral{
			LogN:            14,                                    // A ring degree of 2^{14}
			LogQ:            []int{55, 45, 45, 45, 45, 45, 45, 45}, // An initial prime of 55 bits and 7 primes of 45 bits
			LogP:            []int{61},                             // The log2 size of the key-switching prime
			LogDefaultScale: 45,                                    // The default log2 of the scaling factor
		}); err != nil {
		panic(err)
	}
	prec := params.EncodingPrecision() // we will need this value later

	fmt.Printf("Precision: %d\n", prec)

	// ==============
	// Key Generation
	// ==============
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk) // Note that we can generate any number of public keys associated to the same Secret Key.
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)

	// ====================
	// Plaintext Generation
	// ====================
	LogSlots := params.LogMaxSlots()
	Slots := 1 << LogSlots
	fmt.Printf("Number of slots: %d\n", Slots)

	// We generate a vector of `[]float64` uniformly distributed in [-1, 1]
	/* #nosec G404 -- this is a plaintext vector  */
	r := rand.New(rand.NewSource(0))
	values1 := make([]float64, Slots)
	for i := 0; i < Slots; i++ {
		values1[i] = 2*r.Float64() - 1
	}
	printVec(values1, "values1")

	pt1 := ckks.NewPlaintext(params, params.MaxLevel())
	ecd := ckks.NewEncoder(params)
	ecd2 := ckks.NewEncoder(ckks.Parameters(params))
	if err = ecd2.Encode(values1, pt1); err != nil {
		panic(err)
	}

	// =====================
	// Ciphertext Generation
	// =====================
	enc := rlwe.NewEncryptor(params, pk)
	ct1, err := enc.EncryptNew(pt1)
	if err != nil {
		panic(err)
	}

	// =========
	// Decryptor
	// =========
	dec := rlwe.NewDecryptor(params, sk)

	// ================
	// Evaluator Basics
	// ================
	eval := ckks.NewEvaluator(params, evk)

	// For the purpose of the example, we will create a second vector of random values.
	values2 := make([]float64, Slots)
	for i := 0; i < Slots; i++ {
		values2[i] = 2*r.Float64() - 1
	}
	printVec(values2, "values2")
	pt2 := ckks.NewPlaintext(params, params.MaxLevel())

	fmt.Printf("\n")
	fmt.Printf("========\n")
	fmt.Printf("ADDITION\n")
	fmt.Printf("========\n")
	fmt.Printf("\n")

	if err = ecd.Encode(values2, pt2); err != nil {
		panic(err)
	}

	ct2, err := enc.EncryptNew(pt2)
	if err != nil {
		panic(err)
	}

	// ciphertext + ciphertext
	want := make([]float64, Slots)
	for i := 0; i < Slots; i++ {
		want[i] = values1[i] + values2[i]
	}

	ct3, err := eval.AddNew(ct1, ct2)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Addition - ct + ct%s", ckks.GetPrecisionStats(params, ecd, dec, want, ct3, 0, false).String())

	// decrypts and decode the result
	have, err := decryptDecode(dec, ecd, ct3, params)
	if err != nil {
		panic(err)
	}
	printVec(have, "decrypted")
	printVec(want, "want")
	fmt.Printf("Error: %.4f\n", calculateError(have, want))

	fmt.Printf("\n")
	fmt.Printf("==============\n")
	fmt.Printf("MULTIPLICATION\n")
	fmt.Printf("==============\n")
	fmt.Printf("\n")

	// ciphertext * scalar
	scalar := 0.25
	for i := 0; i < Slots; i++ {
		want[i] = values1[i] * scalar
	}
	ct4, err := eval.MulRelinNew(ct1, scalar)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Multiplication - ct * scalar%s", ckks.GetPrecisionStats(params, ecd, dec, want, ct4, 0, false).String())

	// decrypts and decode the result
	have2, err2 := decryptDecode(dec, ecd, ct4, params)
	if err2 != nil {
		panic(err2)
	}
	printVec(have2, "decrypted")
	printVec(want, "want")
	fmt.Printf("Error: %.4f\n", calculateError(have2, want))

}

func printVec(vec []float64, name string) {
	fmt.Printf("%s: [", name)
	for i := 0; i < 10 && i < len(vec); i++ {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Printf("%.4f", vec[i])
	}
	if len(vec) > 10 {
		fmt.Print(", ...")
	}
	fmt.Println("]")
}

func decryptDecode(
	dec *rlwe.Decryptor,
	ecd *ckks.Encoder,
	ct *rlwe.Ciphertext,
	params ckks.Parameters,
) ([]float64, error) {
	dec_pt := dec.DecryptNew(ct)
	// Decodes the plaintext
	have := make([]float64, params.MaxSlots())
	err := ecd.Decode(dec_pt, have)
	if err != nil {
		return nil, err
	}
	return have, nil
}

func calculateError(have, want []float64) float64 {
	var sum float64
	for i := 0; i < len(have); i++ {
		sum += math.Abs(have[i] - want[i])
	}
	return sum
}
