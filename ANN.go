package main

import (
	"fmt"
	"math"
	"log"
	"math/rand"
	"time"
)

func sigmoid(value float64) float64 {
	return 1 / (1 + math.Exp(-value))
}

func init(){

}

type ANN struct {
	inputs					Layer		//generates on run
	output					Layer
	hiddenLayers		[]Layer
	mutationChance	float64
	mutationRange		float64
	// layerCount			int
}

type ANNInterface interface{
	run([]float64)
	mutateLayers()
	calculateLayers()
	addHiddenLayers([]int)	//generate Layers based on an array of neurons in each layer
	removeHiddenLayer(int)	//remove specific layer
}

func (a *ANN) run(ANNinputs []float64){
	a.inputs.mutationChance = a.mutationChance
	a.inputs.mutationRange = a.mutationRange
	a.inputs.neuronCount = len(ANNinputs)
	a.inputs.inArray = ANNinputs
	a.inputs.generateNeurons()

	for i := 0; i < len(a.hiddenLayers); i++{
		if i == 0{
			a.hiddenLayers[i].inArray = a.inputs.calculateNeurons()
		} else{
			//a.hiddenLayers[i].inArray = a.hiddenLayers[i-1].calculateNeurons()
		}
	}



}

func (a *ANN) addHiddenLayers(array []int){
	for i := 0; i < len(array); i++{
		var l Layer
		//log.Println("Before", l.mutationChance, l.mutationRange)
		l.neuronCount = array[i]
		l.mutationChance = a.mutationChance
		l.mutationRange = a.mutationRange
		l.generateNeurons()
		//log.Println("After", l.mutationChance, l.mutationRange)
		a.hiddenLayers = append (a.hiddenLayers, l)
	}
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

func (l *Layer) print() {
	fmt.Printf("Layer: \n")
	fmt.Printf("  NCount: [%d]: \n", l.neuronCount)
	fmt.Printf("  Mutation Chance: [%f]: \n", l.mutationChance)
	fmt.Printf("  Mutation Range: [%f]: \n", l.mutationRange)

	fmt.Printf("  Input Array: [ ")
	for in:= 0; in < len(l.inArray); in++ {
		fmt.Printf("%f ", l.inArray[in])
	}
	fmt.Printf("]\n")

	fmt.Printf("  Output Array: [ ")
	for out := 0; out < len(l.outArray); out++ {
		fmt.Printf("%f ", l.outArray[out])
	}
	fmt.Printf("]\n")

	for i:= 0; i < len(l.neurons); i++ {
		fmt.Printf("  Neuron: [%d]: \n", i)
		l.neurons[i].print()
	}
}

func (l *Layer) printDotLabels(layer int) {

	// Print the labels for each neuron
	for i:= 0; i < len(l.neurons); i++ {
		fmt.Printf("L%dN%d [label=\"L%dN%d: \\nBias: %f\"]\n", layer, i, layer, i, l.neurons[i].inputBias)
	}
}

func (l *Layer) printDotStructure(layer int) {

	// Print the labels for each neuron
	fmt.Printf("{ rank=same ")
	for i:= 0; i < len(l.neurons); i++ {
		fmt.Printf("L%dN%d ", layer, i)
	}
	fmt.Printf("} ")
}

func (l *Layer) generateNeurons(){
	for i := 0; i < l.neuronCount; i++{
		var n Neuron
		n.generate(len(l.inArray))
		l.neurons = append(l.neurons, n)
	}
}

func (l *Layer) mutateNeurons(){
	for i := 0; i < len(l.neurons); i++{
		l.neurons[i].mutate([2]float64 {l.mutationChance, l.mutationRange})
	}
}

func (l *Layer) calculateNeurons() []float64{
	for i := 0; i < len(l.neurons); i++{
		l.outArray = append(l.outArray, l.neurons[i].calculate(l.inArray))
	}
	return l.outArray
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

func (n *Neuron) print() {
	fmt.Println("\t========================================")
	fmt.Println("\t| Neuron Bias: ", n.inputBias)
	for i:= 0; i < len(n.inputWeights); i++ {
		fmt.Printf("\t| Input Weight: [%d] = %f\n", i, n.inputWeights[i])
	}
	fmt.Println("\t|_______________________________________\n")
}

func (n *Neuron) printDot(layer int) {

	// Print the labels for each neuron
	for i:= 0; i < len(n.inputWeights); i++ {
		fmt.Printf("L%dN%d [label=\"Input %d: \nBias: %f\"]\n", layer, i, i, n.inputWeights[i])
	}
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

	var a ANN
	a.mutationChance = 1.0
	a.mutationRange = 10.0
	a.addHiddenLayers([]int{1,2,3})
	a.run([]float64{1.0, 1.0})
	log.Println("0:",a.hiddenLayers[0])
	log.Println("1:",a.hiddenLayers[1])
	log.Println("2:",a.hiddenLayers[2])

}

//var n Neuron
//n.inputWeights = make([]float64, 5);
//n.generate(5)
//log.Println(n.calculate([]float64{1.0, 1.0, 1.0, 1.0, 1.0}))


	/*var n1 Neuron
	var n2 Neuron
	var n3 Neuron
	n1.generate(5);
	n2.generate(5);
	n3.generate(5);

	lay := Layer{
		mutationChance:	1.0,
		mutationRange: 10.0,
		neuronCount: 6,
		neurons: []Neuron{},
		inArray: []float64{1.0,1.0,1.0,1.0,1.0},
		outArray: []float64{},
	}

	lay.generateNeurons()

	//log.Println(lay.inArray)
	//log.Println(lay.outArray)
	log.Println("Calculating...");
	lay.calculateNeurons()
	lay.print()
	*/
	//lay.mutateNeurons()

<<<<<<< HEAD
	//lay.print()
=======
	lay.mutateNeurons()

	lay.print()

	// ================================================
	// beginning of graph structure:
	fmt.Printf("digraph G { node [shape=box fontsize=8] edge [fontsize=8] \n")

	// First print out labels:
	// run for the input layer
	//inputlayer.printDotLabels(1)

	// run for the output layer
	//outputlayer.printDotLabels(1)

	// loop for each hidden layer:
	lay.printDotLabels(1)


	// Now print out structure:
	// run for input layer:
	//inputlayer.printDotStructure(1)
	fmt.Printf(" -> \n")

	// then loop again for each hidden layer:
	lay.printDotStructure(1)
	fmt.Printf(" -> \n")

	// run for the output layer
	//outputlayer.printDotStructure(1)

	// close it up:
	fmt.Printf("}\n")
	// ================================================
}
>>>>>>> 3f7763a183e866d025da9db0dfabbec36e547805
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
