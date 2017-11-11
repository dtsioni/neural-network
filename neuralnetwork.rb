require 'matrix'
require "./hadamard.rb"

class NeuralNetwork
    # activationFunction
    # derivateActivationFunction
    # weightedInputs - holds the weighted inputs to each layer after a feed forward, for use in back prop
    # weights
    attr_accessor :weights

    # layers is an array which defines the size of each layer of the neural network
    def initialize(layers = [], activationFunction = SIGMOID, derivativeActivationFunction = DERIVATIVE_SIGMOID, startingWeightFunction = INVERSE_SQRT)
        @weights = []
        @activationFunction = activationFunction
        @derivativeActivationFunction = derivativeActivationFunction

        # build weights, an array of weight layers
        layers.each_with_index { |val, index|
            next if index == layers.length - 1

            a = layers[index]
            b = layers[index + 1]

            @weights.push(
                Matrix.build(b, a) {
                    |row, col|
                    startingWeightFunction.call(a)
                }
            )
        }
    end

    # examples should be a hash of two matrices - input and output
    def train(examples)
        avgErr = 0
        n = 0

        examples.each do |example|
            actual = feedForward(example[:input])
            error = meanSquareError(actual, example[:output])
            backPropogation(actual, example[:output])

            avgErr = (avgErr * n + error) / (n + 1)
            n += 1
        end
        puts "Average error: #{avgErr}"
    end

    def feedForward(input)
        @weightedInputs = [input]
        output = mapActivationFunction(input)

        weights.each { |weightLayer|
            output = weightLayer * output
            @weightedInputs.push(output)
            output = mapActivationFunction(output)
        }
        return output
    end

    # actual and expected are matrices
    def backPropogation(actual, expected)
        nextError = (actual - expected).hadamard mapDerivativeActivationFunction(@weightedInputs.pop) # output error
        i = @weights.length - 1

        until i < 0 do
            weightLayer = @weights[i]

            z = @weightedInputs.pop
            a = mapActivationFunction(z)

            error = (weightLayer.t * nextError).hadamard mapDerivativeActivationFunction(z) # hidden error
            deltaW = nextError * a.t

            @weights[i] = weightLayer - deltaW

            nextError = error
            i -= 1
        end
    end

    def to_s
        Marshal.dump(@weights)
    end

    def setWeightsFromMarshaledString(marshaledWeights)
        @weights = Marshal.load(marshaledWeights)
    end

    def mapActivationFunction(m)
        return mapFunction(m, @activationFunction)
    end

    def mapDerivativeActivationFunction(m)
        return mapFunction(m, @derivativeActivationFunction)
    end

    def mapFunction(m, f)
        return m.map{ |e| f.call(e) }
    end

    def meanSquareError(a, b)
        diff = a - b
        diff = diff.hadamard diff

        sum = 0
        i = 0
        diff.each do |n|
            sum += n
            i += 1
        end
        error = sum / i
    end

    # default initial weight function
    INVERSE_SQRT = Proc.new do |x|
        max = 1.0/Math.sqrt(x)
        min = -max
        Random.rand(min..max)
    end

    # default activation function
    SIGMOID = Proc.new do |x|
        1/(1+Math.exp(-x))
    end

    DERIVATIVE_SIGMOID = Proc.new do |x|
        sig = SIGMOID.call(x)
        sig * (1 - sig)
    end
end
