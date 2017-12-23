package com.mantono.kann

import com.mantono.kann.ga.NeuralNetwork
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
}