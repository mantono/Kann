package com.mantono.kann

import kotlinx.coroutines.experimental.delay
import java.util.*
import kotlin.math.max

data class NeuralNetwork(
		private val layers: MutableList<Int>,
		private val weights: MutableList<Double>,
		private val biases: MutableList<Double>,
		private val functions: MutableList<(Double) -> Double>
		)
{
	constructor(vararg layers: Int, seed: Long = randomSeed()): this(
			layers.toMutableList(),
			initWeights(layers.toList(), seed),
			initBiases(layers.toList(), seed),
			initFunctions(layers.toList())
	)

	operator fun set(layer: Int, neuron: Int, func: (Double) -> Double)
	{
		val i: Int = layers.take(layer).sum() + neuron
		functions[i] = func
	}

	private operator fun set(layer: Int, neuron: Int, bias: Double)
	{
		val i: Int = layers.take(layer).sum() + neuron
		biases[i] = bias
	}

	operator fun get(layer: Int, neuron: Int): Double
	{
		val i: Int = layers.take(layer).sum() + neuron
		return biases[i]
	}

	private operator fun set(layer: Int, neuron: Int, weight: Int, value: Double)
	{
		val size: Int = if (layer == 0) 0 else layers[layer - 1]
		val start: Int = layers.take(layer).sum() + (neuron * size)
		val end: Int = start + size
		val validRange: IntRange = start..end
		val relativeIndex: Int = start + weight
		if(relativeIndex !in validRange)
			throw IndexOutOfBoundsException("Index $weight is out of bounds for neuron $neuron in layer $layer which has $size weights")
		weights[relativeIndex] = value
	}

	operator fun get(layer: Int, neuron: Int, weight: Int): Double
	{
		val size: Int = weightsInLayer(layer)
		val start: Int = layers.take(layer).sum() + (neuron * size)
		val end: Int = start + size
		val validRange: IntRange = start until end
		val relativeIndex: Int = start + weight
		if(relativeIndex !in validRange)
			throw IndexOutOfBoundsException("Index $weight is out of bounds for neuron $neuron in layer $layer which has $size weights")
		return weights[relativeIndex]
	}

	tailrec fun input(inputs: Array<Double>, layer: Int = 0): Array<Double>
	{
		if(layer > layers.lastIndex)
			return inputs

		val neurons = layers[layer]

		val output: Array<Double> = (0 until neurons).map { neuron ->
			val neuronWeights: List<Double> = weights(layer, neuron)
			val bias: Double = bias(layer, neuron)
			val function = function(layer, neuron)
			val weightedInput: Double = weightedInput(inputs, neuronWeights, bias)
			function(weightedInput)
		}.toTypedArray()

		return input(output, layer + 1)
	}

	private fun weightedInput(i: Array<Double>, w: List<Double>, b: Double): Double
	{
		return i.asSequence()
				.mapIndexed { index: Int, input: Double -> w[index] * input }
				.sum() + b
	}

	fun weights(layer: Int, neuron: Int): List<Double>
	{
		val size: Int = weightsInLayer(layer)
		val start: Int = layers.take(layer).sum() + (neuron * size)
		val end: Int = start + size
		return weights.slice(start until end)
	}

	fun bias(layer: Int, neuron: Int): Double
	{
		val i: Int = layers.take(layer).sum() + neuron
		return biases[i]
	}

	fun function(layer: Int, neuron: Int): (Double) -> Double
	{
		val i: Int = layers.take(layer).sum() + neuron
		return functions[i]
	}

	fun weightsInLayer(layer: Int): Int = if (layer == 0) 1 else layers[layer - 1]

	tailrec fun train(
			data: List<TrainingData>,
			iterations: Int = 1000,
			maxError: Double = 0.1
	): Double
	{
		val dataPointIndex: Int = Random().nextInt(data.size)
		val dataPoint: TrainingData = data[dataPointIndex]
		val predicted: Array<Double> = input(dataPoint.input)
		val result = ResultData(predicted, dataPoint.output)
		val cost: Double = result.squaredError
		val slope: Double = result.slope
		val learningRate = 0.01

		layers.mapIndexed { layer, size ->
			(0 until size).map { neuron ->
				val nFunc = function(layer, neuron)
				val nWeights = weights(layer, neuron)
				val nBias = bias(layer, neuron)

				val weightedInput: Double = weightedInput(dataPoint.input, nWeights, nBias)
				val derivativeFunc: Double = derivative(weightedInput, nFunc)

				nWeights.indices.map { index ->
					dataPoint.input.map { d ->
						val derivativeCostWeight: Double = slope * derivativeFunc * d
						this[layer, neuron, index] += learningRate * derivativeCostWeight
					}
				}

				val derivativeCostBias: Double = slope * derivativeFunc
				this[layer, neuron] += learningRate * derivativeCostBias
			}
		}

		if(iterations % 100 == 0)
		{
			val sumOfCost: Double = evaluate(this, data)
			println(sumOfCost)
			if(sumOfCost <= maxError || iterations == 0) return sumOfCost
		}

		Thread.sleep(10)

		return train(data, iterations - 1, maxError)
	}
}

private fun initWeights(layers: List<Int>, seed: Long): MutableList<Double>
{
	val weightCardinality: Int = layers.mapIndexed { index, size ->
		when(index)
		{
			0 -> size
			else -> layers[index - 1] * size
		}
	}.sum()

	return randomSequence(seed, deterministic = true).take(weightCardinality).toMutableList()
}

private fun initFunctions(layers: List<Int>): MutableList<(Double) -> Double>
{
	return Array<(Double) -> Double>(layers.sum(), { _ -> ::sigmoid }).toMutableList()
}

private fun initBiases(layers: List<Int>, seed: Long): MutableList<Double>
{
	val biasesCardinality: Int = layers.sum()
	return randomSequence(seed * 911, deterministic = true).take(biasesCardinality).toMutableList()
}


// OLD HERE


//
//data class NeuralNetwork(private val neurons: MutableList<MutableList<Neuron>>)
//{
//	constructor(vararg layers: Int, seed: Long = randomSeed()): this(listOfLayers(layers, seed))
//
//	val size: Int = neurons.map { it.size }.sum()
//
//	operator fun get(layer: Int): List<Neuron> = neurons[layer]
//	operator fun get(layer: Int, index: Int): Neuron = neurons[layer][index]
//	operator fun get(layer: Int, neuron: Int, weight: Int): Double = neurons[layer][neuron][weight]
//
//	private operator fun set(layer: Int, neuron: Int, weight: Int, value: Double)
//	{
//		neurons[layer][neuron][weight] = value
//	}
//
//	operator fun set(layer: Int, neuron: Int, func: (Double) -> Double)
//	{
//		neurons[layer][neuron] = neurons[layer][neuron].copy(function = func)
//	}
//
//	fun getInBounds(index: Int): Double
//	{
//		val layer: List<Neuron> = this[(index * 7) % size]
//		val neuron: Neuron = layer[(index * 11) % layer.size]
//		return neuron[(index * 31) % neuron.size]
//	}
//
//	fun modifyInBounds(index: Int, change: Double): NeuralNetwork
//	{
//		return when(index % 19 == 0)
//		{
//			true -> modifyBias(index, change)
//			false -> modifyWeight(index, change)
//		}
//	}
//
//	private fun modifyWeight(index: Int, change: Double): NeuralNetwork
//	{
//		val layerIndex = (index * 7) % neurons.size
//		val layer: List<Neuron> = this[layerIndex]
//		val neuronIndex: Int = (index * 11) % layer.size
//		val neuron: Neuron = layer[neuronIndex]
//		val weightIndex = (index * 31) % neuron.size
//
//		val current: Double = neurons[layerIndex][neuronIndex][weightIndex]
//
//		val newLayers: MutableList<MutableList<Neuron>> = neurons.map { it.toMutableList() }.toMutableList()
//		val newWeight: Double = current + change
//		val newNeuron: Neuron = newLayers[layerIndex][neuronIndex].set(weightIndex, newWeight)
//		newLayers[layerIndex][neuronIndex] = newNeuron
//
//		return NeuralNetwork(newLayers)
//	}
//
//	private fun modifyBias(index: Int, change: Double): NeuralNetwork
//	{
//		val layerIndex = (index * 7) % neurons.size
//		val layer: List<Neuron> = this[layerIndex]
//		val neuronIndex: Int = (index * 11) % layer.size
//		val neuron: Neuron = layer[neuronIndex]
//
//		val current: Double = neurons[layerIndex][neuronIndex].bias
//
//		val newLayers: MutableList<MutableList<Neuron>> = neurons.map { it.toMutableList() }.toMutableList()
//		val newBias: Double = current + change
//		val newNeuron = Neuron(neuron.weights().toMutableList(), newBias, neuron.function)
//		newLayers[layerIndex][neuronIndex] = newNeuron
//
//		return NeuralNetwork(newLayers)
//	}
//
////	fun add(layer: Int, neuron: Neuron)
////	{
////		neurons[layer].add(neuron)
////	}
//
//	tailrec fun input(inputs: Array<Double>, layer: Int = 0): Array<Double>
//	{
//		if(layer > neurons.lastIndex)
//			return inputs
//
//		val output: Array<Double> = neurons[layer]
//				.map { it.feedInput(inputs) }
//				.toTypedArray()
//
//		return input(output, layer + 1)
//	}
//
//	fun mutate(mutationFactor: Double = 0.1): NeuralNetwork
//	{
//		val mutatedNeurons: MutableList<MutableList<Neuron>> = neurons.map { layer ->
//			layer.map { it.mutate(mutationFactor) }.toMutableList()
//		}
//				.toMutableList()
//
//		return NeuralNetwork(mutatedNeurons)
//	}
//
//	tailrec fun train(
//			data: List<TrainingData>,
//			iterations: Int = 1000,
//			maxError: Double = 0.1,
//			nn: NeuralNetwork = this,
//			previousSumOfCost: Double = Double.MAX_VALUE
//	): Pair<NeuralNetwork, Int>
//	{
//		if(iterations == 0) return nn to 0
//
//		val dataPointIndex: Int = iterations % data.size
//		val dataPoint: TrainingData = data[dataPointIndex]
//		val predicted: Array<Double> = nn.input(dataPoint.input)
//		val result = ResultData(predicted, dataPoint.output)
//		val cost: Double = result.squaredError
//		val slope: Double = result.slope
//		val learningRate = 0.01
//
//		val adjusted: MutableList<MutableList<Neuron>> = nn.neurons.map { layer ->
//			layer.map { neuron ->
//				neuron.f
//
//				val derivativeFunc: Double = derivative(neuron, dataPoint.input)
//				val adjustedWeights: List<Double> = neuron.weights().mapIndexed { i, weight ->
//					val derivCostWeight: Double = slope * derivativeFunc * i // Use the dataPoint input as "i"
//					val changedWeight = weight - (learningRate * derivCostWeight)
//					changedWeight
//				}.toList()
//				val derivCostBias: Double = slope * derivativeFunc
//				val adjustedBias: Double = neuron.bias - (learningRate * derivCostBias)
//
//				//Neuron(adjustedWeights, adjustedBias, neuron.function)
//			}.toList()
//		}.toList()
//
//		val adjustedNetwork = NeuralNetwork(adjusted)
//
//		val sumOfCost: Double = evaluate(adjustedNetwork, data)
//
//		if(iterations % 100 == 0 || sumOfCost < previousSumOfCost)
//			println(sumOfCost)
//
//		if(sumOfCost <= maxError) return nn to iterations
//
//		return train(data, iterations - 1, maxError, adjustedNetwork, sumOfCost)
//	}
//
//	override fun hashCode(): Int
//	{
//		return neurons.map {
//			it.map { it.hashCode() }
//					.reduce { acc, i -> acc xor i }
//		}
//				.sum()
//	}
//
//	override fun toString(): String
//	{
//		val str = StringBuilder("network {\n")
//		str.append("\tlistOf(\n")
//		neurons.forEach {list ->
//			val n = list.joinToString(prefix = "\t\tlistOf(", separator = ", ", postfix = "),\n")
//			str.append(n)
//		}
//		str.delete(str.lastIndex-1, str.lastIndex)
//		str.append("\t)")
//		str.append("\n}")
//		return str.toString()
//	}
//}

//fun network(layers: () -> List<List<Neuron>>): NeuralNetwork
//{
//	val mute: MutableList<Neuron> = layers.invoke().map { it.toMutableList() }
//	return NeuralNetwork(neurons = mute)
//}

fun evaluate(n: NeuralNetwork, data: Collection<TrainingData>): Double
{
	return data
			.map { trainingEntry ->
				val output: Array<Double> = n.input(trainingEntry.input)
				ResultData(output, trainingEntry.output)
			}
			.map { it.squaredError }
			.sum()
}

fun averageSlope(n: NeuralNetwork, data: Collection<TrainingData>): Double
{
	return data
			.map { trainingEntry ->
				val output: Array<Double> = n.input(trainingEntry.input)
				ResultData(output, trainingEntry.output)
			}
			.map { it.slope }
			.average()
}

private fun listOfLayers(layers: IntArray, seed: Long): MutableList<MutableList<Neuron>>
{
	return layers
			.mapIndexed { layer: Int, neuronInLayer: Int ->
				Array(neuronInLayer) { index: Int ->
					val uniqueSeed: Long = generateSeedFrom(seed, layer, neuronInLayer, index)
					val connectionsNextLayer = layers[max(layer - 1, 0)]
					Neuron(connectionsNextLayer, randomSequence(uniqueSeed))
				}
			}
			.map { it.toMutableList() }
			.toMutableList()
}

fun layerOf(vararg funcs: (Double) -> Double): List<Neuron>
{
	return funcs.map { Neuron(function = it) }.toList()
}

private const val DELTA_X: Double = 0.000000001

fun derivative(weightedInputs: Double, func: (Double) -> Double): Double
{
	val y1: Double = func(weightedInputs)
	val y2: Double = func(weightedInputs + DELTA_X)
	val deltaY = y2 - y1
	return deltaY / DELTA_X
}