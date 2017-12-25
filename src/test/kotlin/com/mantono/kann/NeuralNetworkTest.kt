package com.mantono.kann

import com.mantono.kann.ga.NeuralNetwork
import com.mantono.kann.ga.Neuron
import com.mantono.kann.ga.TrainingData
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class NeuralNetworkTest
{
	@Test
	fun creationTestWithOnlyLayersAndSeedConstructor()
	{
		val nn = NeuralNetwork(3, 2, 2, seed =  2L)
		println(nn)
		val result = nn.input(arrayOf(2.0, 3.0, 4.0))
		result.forEach { print("$it, ") }
	}

	@Test
	fun trainSingleLayerSingleInputNetworkLinearFunctionTest()
	{
		val neuron = Neuron(arrayOf(0.25), 0.5) { it }
		val nn = NeuralNetwork(listOf(listOf(neuron)))
		val data = listOf(
				TrainingData(1.0, 2.0),
				TrainingData(2.0, 4.0),
				TrainingData(4.0, 8.0)
		)

		val iterations = 100_000
		val final = nn.train(data, iterations, 0.001)
		println(final.first)
		println(iterations - final.second)

		assertTrue(final.second > 0)
	}

	@Test
	fun trainSingleLayerSingleInputNetworkNonLinearFunctionTest()
	{
		val neuron = Neuron(arrayOf(0.25), 0.5) { it }
		val nn = NeuralNetwork(listOf(listOf(neuron)))
		val data = listOf(
				TrainingData(1.0, 2.0),
				TrainingData(2.0, 4.0),
				TrainingData(4.0, 5.0)
		)

		val iterations = 100_000
		val final = nn.train(data, iterations, 0.642858)
		println(final.first)
		println(iterations - final.second)

		assertTrue(final.second > 0)
	}

	@Test
	fun trainSingleLayerSingleInputNetworkExponentialFunctionTest()
	{
		val neuron = Neuron(arrayOf(0.25), 0.5) { Math.pow(it, 2.0) }
		val nn = NeuralNetwork(listOf(listOf(neuron)))
		val data = listOf(
				TrainingData(2.0, 4.0),
				TrainingData(4.0, 16.0),
				TrainingData(8.0, 64.0)
		)

		val iterations = 100_000
		val final = nn.train(data, iterations, 0.001)
		println(final.first)
		println(iterations - final.second)

		assertTrue(final.second > 0)
	}
}