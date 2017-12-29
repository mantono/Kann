package com.mantono.kann

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.io.File
import java.nio.charset.Charset
import java.nio.file.Files

private val data = Files.readAllLines(File("src/test/kotlin/com/mantono/kann/test_data.csv").toPath(), Charset.forName("UTF-8"))
		.asSequence()
		.map { it.split(",") }
		.map { it[0].toDouble() to it[1].toDouble() }
		.map { TrainingData(it.first, it.second) }
		.toList()

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
	fun testTrainStochastic()
	{
		val nn = NeuralNetwork(
				layers = mutableListOf(1, 4, 3, 1),
				weights = mutableListOf(-18.97991626395286, -1.886891949541143, 0.005581196633805963, -0.896901487194891, -15.587804502819369, -553.4989234838522, 225.97174568487168, -66.63154243283833, -237.39099905587838, 303.70244692576387, -1.818441520293618, 15.101033054162977, -10.607822687543338, 39.52225377616854, -1087.372955142125, -1321.0074696901581, 1580.0650252629814, 0.0779282468563818, -0.8353075496349948, -1.2060745319672839),
				biases = mutableListOf(112.20035513629327, -3.053779918377, 13.588609728214617, -2.5647177843207114, 14.990172709664158, 635.9517538094807, 51.06936802676885, 617.506876337792, -375.19616910419796)
		)

		nn[0, 0] = { it }
		nn[2, 2] = { it }
		nn[3, 0] = { it }
		val final = nn.trainStochastic(data, 10_000_000)
		println(final)

		assertEquals(2.275, final.input(arrayOf(80.0))[0], 0.2)
		assertEquals(2.5625, final.input(arrayOf(75.0))[0], 0.3)
		assertEquals(3.8333, final.input(arrayOf(70.0))[0], 0.23)
	}

	@Test
	fun testTrainBatch()
	{
		val nn = NeuralNetwork(
				layers = mutableListOf(1, 4, 3, 1),
				weights = mutableListOf(-0.24012116844313372, 0.7277312994623772, -0.28984808108066096, -0.2808544042666866, -0.829211559493815, 2.294746246383296, -0.17482740017203574, -3.3834261187711356, 1.328281262096718, 0.8269305445719277, -1.3668840501483641, -0.014117249480521288, 0.7614336355759709, -4.043499516793906, -0.0528212256508564, 1.6151405308079363, -1.6347757113423984, -0.7113577708626267, -1.0866726301400216, -0.5396487695071697),
				biases = mutableListOf(5.395985981904578, 5.383295343447355, -500.9488726294248, -3.4770894570965623, -3.06941282171144, -0.5206355010228028, -0.8737766643800498, -0.524802590095036, 0.27371375582880925)
		)

		nn[0, 0] = { it }
		nn[2, 2] = { it }
		nn[2, 1] = { Math.log1p(it.coerceAtLeast(0.0)) }
		nn[3, 0] = { it }
		val final = nn.trainBatch(data, 10_000)
		println(final)

		assertEquals(2.275, final.input(arrayOf(80.0))[0], 0.2)
		assertEquals(2.5625, final.input(arrayOf(75.0))[0], 0.3)
		assertEquals(3.8333, final.input(arrayOf(70.0))[0], 0.23)
	}

//	@Test
//	fun trainSingleLayerSingleInputNetworkLinearFunctionTest()
//	{
//		val neuron = Neuron(listOf(0.25), 0.5) { it }
//		val nn = NeuralNetwork(listOf(listOf(neuron)))
//		val data = listOf(
//				TrainingData(1.0, 2.0),
//				TrainingData(2.0, 4.0),
//				TrainingData(4.0, 8.0)
//		)
//
//		val iterations = 100_000
//		val final = nn.train(data, iterations, 0.001)
//		println(final.first)
//		println(iterations - final.second)
//
//		assertTrue(final.second > 0)
//	}
//
//	@Test
//	fun trainSingleLayerSingleInputNetworkNonLinearFunctionTest()
//	{
//		val neuron = Neuron(listOf(0.25), 0.5) { it }
//		val nn = NeuralNetwork(listOf(listOf(neuron)))
//		val data = listOf(
//				TrainingData(1.0, 2.0),
//				TrainingData(2.0, 4.0),
//				TrainingData(4.0, 5.0)
//		)
//
//		val iterations = 100_000
//		val final = nn.train(data, iterations, 0.642858)
//		println(final.first)
//		println(iterations - final.second)
//
//		assertTrue(final.second > 0)
//	}
//
//	@Test
//	fun trainSingleLayerSingleInputNetworkExponentialFunctionTest()
//	{
//		val neuron = Neuron(listOf(0.25), 0.5) { Math.pow(it, 2.0) }
//		val nn = NeuralNetwork(listOf(listOf(neuron)))
//		val data = listOf(
//				TrainingData(2.0, 4.0),
//				TrainingData(4.0, 16.0),
//				TrainingData(8.0, 64.0)
//		)
//
//		val iterations = 1_000_000
//		val final = nn.train(data, iterations, 0.001)
//		println(final.first)
//		println(iterations - final.second)
//
//		assertTrue(final.second > 0)
//	}
//
//	@Test
//	fun smallTest()
//	{
//		val nn = network {
//			listOf(
//					listOf(Neuron(weights = listOf(-0.47206288534192786), bias = 72.73833930738654, function = { it })),
//					listOf(Neuron(weights = listOf(1.5241860509638594), bias = 0.6737336875259039, function = { it }), Neuron(weights = listOf(18.293722354183764), bias = -0.0041484554450079035, function = { Math.log1p(it.coerceAtLeast(0.0)) }), Neuron(weights = listOf(2.2651915399663896), bias = -3.741487749113519, function = ::sigmoid), Neuron(weights = listOf(78.1388839150298), bias = 0.0, function = ::sigmoid)),
//					listOf(Neuron(weights = listOf(0.2045802868342827, 0.25919862194264703, -9.978705704409432, 0.24073069545967418), bias = -0.7758684412486735, function = { it.coerceAtLeast(0.0) }))
//			)
//		}
//
//
//		val inputLayer = listOf(Neuron(weights = listOf(-0.5759733120664398), bias = 67.56638566038487, function = { it }))
//		val secondLayer = layerOf({ it }, { Math.log1p(it) }, ::sigmoid, ::sigmoid)
//		val outputLayer = layerOf({ it.coerceAtLeast(0.0) })
//
//
//		val data = Files.readAllLines(File("src/test/kotlin/com/mantono/kann/test_data.csv").toPath(), Charset.forName("UTF-8"))
//				.asSequence()
//				.map { it.split(",") }
//				.map { it[0].toDouble() to it[1].toDouble() }
//				.map { TrainingData(it.first, it.second) }
//				.toList()
//
//		val iterations = 100_000
//		val final = nn.train(data, iterations, 0.01)
//		println(final.first)
//		assertEquals(2.275, final.first.input(arrayOf(80.0))[0], 0.2)
//		assertEquals(2.5625, final.first.input(arrayOf(75.0))[0], 0.3)
//		assertEquals(3.8333, final.first.input(arrayOf(70.0))[0], 0.23)
//	}
}