package com.mantono.kann

import com.mantono.kann.ga.NeuralNetwork
import com.mantono.kann.ga.Neuron
import com.mantono.kann.ga.TrainingData
import com.mantono.kann.ga.layerOf
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.io.File
import java.nio.charset.Charset
import java.nio.file.Files

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
		val neuron = Neuron(listOf(0.25), 0.5) { it }
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
		val neuron = Neuron(listOf(0.25), 0.5) { it }
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
		val neuron = Neuron(listOf(0.25), 0.5) { Math.pow(it, 2.0) }
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

	@Test
	fun smallTest()
	{
		val inputLayer = layerOf({ it })
		val secondLayer = layerOf({ it }, { it * it })
		val thirdLayer = layerOf({ Math.abs(Math.pow(it, 3.0)) }, ::sigmoid)
		val outputLayer = layerOf({ it })

		val nn = NeuralNetwork(listOf(inputLayer, secondLayer, thirdLayer, outputLayer))
		val data = Files.readAllLines(File("src/test/kotlin/com/mantono/kann/test_data.csv").toPath(), Charset.forName("UTF-8"))
				.asSequence()
				.map { it.split(",") }
				.map { it[0].toDouble() to it[1].toDouble() }
				.map { TrainingData(it.first, it.second) }
				.toList()

		val iterations = 10_000
		val final = nn.train(data, iterations, 0.01)
		println(final.first)
		println(iterations - final.second)

		assertTrue(final.second > 0)
	}
}