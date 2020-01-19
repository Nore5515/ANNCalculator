package main

import (
	"math"
	"log"
	"math/rand"
	"time"
	/*"fmt"*/
)

func sigmoid(value float64) float64 {
	return 1 / (1 + math.Exp(-value))
}

func init(){

}

type Layer struct {
	mutationChance	float64
	mutationRange		float64
	neuronCount			int
	neurons					[]Neuron
	inArray					[]float64
	outArray				[]float64
}

type LayerInterface interface{
	generateNeurons()				//create neuronCount neruons to add to the array
	mutateNeurons()		//issue the command to mutate all nodes
	calculateNeurons()	//issue the command to calculate all nodes
}

func (l *Layer) generateNeurons(){
	var array []Neuron
	for i := 0; i < l.neuronCount; i++{
		var n Neuron
		n.generate(len(l.inArray))
		array = append(array, n)
	}
}

func (l *Layer) mutateNeurons(){
	for i := 0; i < len(l.neurons); i++{
		l.neurons[i].mutate([2]float64 {l.mutationChance, l.mutationRange})
	}
}

func (l *Layer) calculateNeurons(){
	for i := 0; i < len(l.neurons); i++{
		l.outArray = append(l.outArray, l.neurons[i].calculate(l.inArray))
	}
}

// A Neuron is a single element within an ANN
type Neuron struct {
	inputBias        float64
	inputWeights		 []float64
}

type NeuronInterface interface {
	mutate([2]float64)	//chance of mutation followed by mutation range (between 0 and 1)
	// this returns the sigmoid'd output of the neuron, given the input array
	calculate([]float64) float64
	generate(int)
}

func (n *Neuron) generate (size int){
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	n.inputWeights = make([]float64, size)
	for i := 0; i < size; i++{
		n.inputWeights[i] = r.Float64();
	}
	n.inputBias = 0.0
}

//for each weight, roll mutation chance. on roll success, change the value by +/- mutation range
func (n *Neuron) mutate (stuff [2]float64) {	//for stuff, it's always an array of 2 values; mutation chance and range
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := 0; i < len(n.inputWeights); i++{
		if (r.Float64() < stuff[0]){
			n.inputWeights[i] = sigmoid(n.inputWeights[i] * (r.Float64() * stuff[1]))
		}
	}
}

func (n *Neuron) calculate (inputArray []float64) float64 {
	x := 0.0;	//total sum of all input*weight combos
	//log.Println(inputArray);
	//log.Println(n.inputWeights);
	for i := 0; i < len(inputArray); i++ {
		//log.Print(i);
		x += inputArray[i] * n.inputWeights[i];
	}
	x += n.inputBias;
	x = sigmoid(x);
	return x
}

func main(){
	var n Neuron
	//n.inputWeights = make([]float64, 5);
	n.generate(5)
	//log.Println(n.calculate([]float64{1.0, 1.0, 1.0, 1.0, 1.0}))

	var n1 Neuron
	var n2 Neuron
	var n3 Neuron
	n1.generate(5);
	n2.generate(5);
	n3.generate(5);

	lay := Layer{
		mutationChance:	1.0,
		mutationRange: 10.0,
		neuronCount: 6,
		neurons: []Neuron{n1,n2,n3},
		inArray: []float64{1.0,1.0,1.0,1.0,1.0},
		outArray: []float64{},
	}

	log.Println(lay.inArray)
	//log.Println(lay.outArray)
	log.Println("Calculating...");
	lay.calculateNeurons()
	log.Println(lay.outArray)
}




/*	var n1 Neuron
	var n2 Neuron
	var n3 Neuron

	var l Layer //{mutationChance:1.0, mutationRange:10.0, neuronCount:6, neurons:[0]Neuron, inArray:[0]float64, outArray:[0]float64}
	lay := Layer{
		mutationChance:	1.0,
		mutationRange: 10.0,
		neuronCount: 6,
		neurons: []Neuron{n1,n2,n3},
	}
	log.Println(l, lay)
*/




	/*n.mutate([2]float64{1.0,10.0})
	log.Println(n.calculate([]float64{1.0}))
	n.mutate([2]float64{1.0,10.0})
	log.Println(n.calculate([]float64{1.0}))
	n.mutate([2]float64{1.0,10.0})
	log.Println(n.calculate([]float64{1.0}))
	*/
	//r := rand.New(rand.NewSource(time.Now().UnixNano()))
	//log.Println(r.Float64())
	//a good example of the range of the sigmoid function
	//log.Println(sigmoid(-10.0), sigmoid (-1.0), sigmoid (-0.5), sigmoid (0.0), sigmoid (0.5), sigmoid (1.0), sigmoid (10.0))
