package com.mantono.kann

import kotlin.math.max

data class NeuralNetwork(private val neurons: List<List<Neuron>>)
{
	constructor(vararg layers: Int, seed: Long): this(listOfLayers(layers, seed))

	val size: Int = neurons.map { it.size }.sum()

	operator fun get(layer: Int): List<Neuron> = neurons[layer]
	operator fun get(layer: Int, index: Int): Neuron = neurons[layer][index]
	operator fun get(layer: Int, neuron: Int, weight: Int): Double = neurons[layer][neuron][weight]
//	operator fun set(layer: Int, index: Int, neuron: Neuron)
//	{
//		neurons[layer][index] = neuron
//	}

	fun getInBounds(index: Int): Double
	{
		val layer: List<Neuron> = this[(index * 7) % size]
		val neuron: Neuron = layer[(index * 11) % layer.size]
		return neuron[(index * 31) % neuron.size]
	}

	fun modifyInBounds(index: Int, change: Double): NeuralNetwork
	{
		return when(index % 19 == 0)
		{
			true -> modifyBias(index, change)
			false -> modifyWeight(index, change)
		}
	}

	private fun modifyWeight(index: Int, change: Double): NeuralNetwork
	{
		val layerIndex = (index * 7) % neurons.size
		val layer: List<Neuron> = this[layerIndex]
		val neuronIndex: Int = (index * 11) % layer.size
		val neuron: Neuron = layer[neuronIndex]
		val weightIndex = (index * 31) % neuron.size

		val current: Double = neurons[layerIndex][neuronIndex][weightIndex]

		val newLayers: MutableList<MutableList<Neuron>> = neurons.map { it.toMutableList() }.toMutableList()
		val newWeight: Double = current + change
		val newNeuron: Neuron = newLayers[layerIndex][neuronIndex].set(weightIndex, newWeight)
		newLayers[layerIndex][neuronIndex] = newNeuron

		return NeuralNetwork(newLayers)
	}

	private fun modifyBias(index: Int, change: Double): NeuralNetwork
	{
		val layerIndex = (index * 7) % neurons.size
		val layer: List<Neuron> = this[layerIndex]
		val neuronIndex: Int = (index * 11) % layer.size
		val neuron: Neuron = layer[neuronIndex]

		val current: Double = neurons[layerIndex][neuronIndex].bias

		val newLayers: MutableList<MutableList<Neuron>> = neurons.map { it.toMutableList() }.toMutableList()
		val newBias: Double = current + change
		val newNeuron = Neuron(neuron.weights().toMutableList(), newBias, neuron.function)
		newLayers[layerIndex][neuronIndex] = newNeuron

		return NeuralNetwork(newLayers)
	}

//	fun add(layer: Int, neuron: Neuron)
//	{
//		neurons[layer].add(neuron)
//	}

	tailrec fun input(inputs: Array<Double>, layer: Int = 0): Array<Double>
	{
		if(layer > neurons.lastIndex)
			return inputs

		val output: Array<Double> = neurons[layer]
				.map { it.feedInput(inputs) }
				.toTypedArray()

		return input(output, layer + 1)
	}

	fun mutate(mutationFactor: Double = 0.1): NeuralNetwork
	{
		val mutatedNeurons: List<MutableList<Neuron>> = neurons.map { layer ->
			layer.map { it.mutate(mutationFactor) }.toMutableList()
		}
				.toList()

		return NeuralNetwork(mutatedNeurons)
	}

	tailrec fun train(
			data: Collection<TrainingData>,
			iterations: Int = 1000,
			maxError: Double = 0.1,
			nn: NeuralNetwork = this,
			attempts: Int = 0
	): Pair<NeuralNetwork, Int>
	{
		if(iterations == 0) return nn to 0

		val resultOriginal: Double = evaluate(nn, data)
		val step: Double = averageSlope(nn, data) * 10 / (1 + attempts * 2)

		if(attempts % 100 == 0)
			println(resultOriginal)

		if(resultOriginal <= maxError) return nn to iterations

		val less: NeuralNetwork = nn.modifyInBounds(iterations, - step)
		val more: NeuralNetwork = nn.modifyInBounds(iterations, step)

		val resultLess = evaluate(less, data)
		val resultMore = evaluate(more, data)

		val bestFit: NeuralNetwork = sequenceOf(
				resultOriginal to nn,
				resultLess to less,
				resultMore to more
		)
				.sortedBy { it.first }
				.map { it.second }
				.first()

		return when(bestFit === nn)
		{
			true -> train(data, iterations - 1, maxError, bestFit, attempts + 1)
			false -> train(data, iterations - 1, maxError, bestFit, 0)
		}
	}

	override fun hashCode(): Int
	{
		return neurons.map {
			it.map { it.hashCode() }
					.reduce { acc, i -> acc xor i }
		}
				.sum()
	}

	override fun toString(): String
	{
		val str = StringBuilder("network {\n")
		str.append("\tlistOf(\n")
		neurons.forEach {list ->
			val n = list.joinToString(prefix = "\t\tlistOf(", separator = ", ", postfix = "),\n")
			str.append(n)
		}
		str.delete(str.lastIndex-1, str.lastIndex)
		str.append("\t)")
		str.append("\n}")
		return str.toString()
	}
}

fun network(layers: () -> List<List<Neuron>>): NeuralNetwork = NeuralNetwork(neurons = layers.invoke())

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

private fun listOfLayers(layers: IntArray, seed: Long): List<MutableList<Neuron>>
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
}

fun layerOf(vararg funcs: (Double) -> Double): List<Neuron>
{
	return funcs.map { Neuron(function = it) }.toList()
}