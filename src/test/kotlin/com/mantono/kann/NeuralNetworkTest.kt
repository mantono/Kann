package com.mantono.kann

import org.junit.jupiter.api.Assertions.assertEquals
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
		val nn = NeuralNetwork(3, 2, 2, seed = 2L)
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
		val nn = network {
			listOf(
					listOf(Neuron(weights = listOf(-0.47206288534192786), bias = 72.73833930738654, function = { it })),
					listOf(Neuron(weights = listOf(1.5241860509638594), bias = 0.6737336875259039, function = { it }), Neuron(weights = listOf(18.293722354183764), bias = -0.0041484554450079035, function = { Math.log1p(it) }), Neuron(weights = listOf(2.2651915399663896), bias = -3.741487749113519, function = ::sigmoid), Neuron(weights = listOf(78.1388839150298), bias = 0.0, function = ::sigmoid)),
					listOf(Neuron(weights = listOf(0.2045802868342827, 0.25919862194264703, -9.978705704409432, 0.24073069545967418), bias = -0.7758684412486735, function = { it.coerceAtLeast(0.0) }))
			)
		}


		val inputLayer = listOf(Neuron(weights = listOf(-0.5759733120664398), bias = 67.56638566038487, function = { it }))
		val secondLayer = layerOf({ it }, { Math.log1p(it) }, ::sigmoid, ::sigmoid)
		val outputLayer = layerOf({ it.coerceAtLeast(0.0) })


		val data = Files.readAllLines(File("src/test/kotlin/com/mantono/kann/test_data.csv").toPath(), Charset.forName("UTF-8"))
				.asSequence()
				.map { it.split(",") }
				.map { it[0].toDouble() to it[1].toDouble() }
				.map { TrainingData(it.first, it.second) }
				.toList()

		val iterations = 100
		val final = nn.train(data, iterations, 0.01)
		println(final.first)
		assertEquals(2.275, final.first.input(arrayOf(80.0))[0], 0.2)
		assertEquals(2.5625, final.first.input(arrayOf(75.0))[0], 0.3)
		assertEquals(3.8333, final.first.input(arrayOf(70.0))[0], 0.23)
	}
}