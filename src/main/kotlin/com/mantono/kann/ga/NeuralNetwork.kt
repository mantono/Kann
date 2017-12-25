package com.mantono.kann.ga

import com.mantono.kann.averageOfSquaredErrorCost
import com.mantono.kann.generateSeedFrom
import com.mantono.kann.randomSeed
import com.mantono.kann.randomSequence
import com.mantono.kann.slope
import com.mantono.kann.transform
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
		val layerIndex = (index * 7) % size
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
		val layerIndex = (index * 7) % size
		val layer: List<Neuron> = this[layerIndex]
		val neuronIndex: Int = (index * 11) % layer.size
		val neuron: Neuron = layer[neuronIndex]

		val current: Double = neurons[layerIndex][neuronIndex].bias

		val newLayers: MutableList<MutableList<Neuron>> = neurons.map { it.toMutableList() }.toMutableList()
		val newBias: Double = current + change
		val newNeuron = Neuron(neuron.weights, newBias, neuron.function)
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
			nn: NeuralNetwork = this
	): Pair<NeuralNetwork, Int>
	{
		if(iterations == 0) return nn to 0

		val resultOriginal: Double = evaluate(nn, data)

		println(resultOriginal)

		if(resultOriginal <= maxError) return nn to iterations

		val primary = nn.modifyBias(iterations, 0.001)
		val resultPrimary = evaluate(primary, data)
		val step = (resultPrimary - resultOriginal) / 0.001

		val less: NeuralNetwork = nn.modifyInBounds(iterations, - step * 0.001)
		val more: NeuralNetwork = nn.modifyInBounds(iterations, step * 0.001)

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

		return train(data, iterations - 1, maxError, bestFit)
	}

	override fun toString(): String
	{
		return neurons.joinToString(separator = "\n", prefix = "\n{\n", postfix = "\n}") { "\t${it.size}: $it" }
	}

	override fun hashCode(): Int
	{
		return neurons.map {
			it.map { it.hashCode() }
					.reduce { acc, i -> acc xor i }
		}
				.sum()
	}
}

fun evaluate(n: NeuralNetwork, data: Collection<TrainingData>): Double
{
	return data
			.map { trainingEntry ->
				val output: Array<Double> = n.input(trainingEntry.input)
				ResultData(output, trainingEntry.output)
			}
			.map { it.squaredError }
			.average()
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