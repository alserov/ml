package main

import "fmt"

func main() {
	// dot product
	// works only if len(a) == len(b)
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	// z = a[0] * b[0] + a[1] * b[1] ... + a[n] * b[n] == dotProduct(a,b)
	// the same as 'countOutput() (3cecd243387fcb7b06a3c2676ebe6ba72d51314c)' without adding bias
	fmt.Println(dotProduct(a, b))
}

func dotProduct(a, b []float32) float32 {
	var res float32
	for i, v := range a {
		res += v * b[i]
	}
	return res
}
