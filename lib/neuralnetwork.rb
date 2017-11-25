require 'matrix'

class NeuralNetwork
    # activationFunction
    # derivateActivationFunction
    # weightedInputs - holds the weighted inputs to each layer after a feed forward, for use in back prop
    # weights
    # biases - holds the bias of each node in the network
    # learning rate - scalar which affects how much a weight changes
    attr_accessor :weights

    # required: layers
    def initialize(args)
        args = defaults.merge(args)
        layers = args[:layers]
        @activationFunction = args[:activationFunction]
        @derivativeActivationFunction = args[:derivativeActivationFunction]
        @learningRate = args[:learningRate]


        @weights = []
        @biases = []
        # build weights, an array of weight layers
        layers.each_with_index { |val, index|
            @biases.push(Matrix.columns([Array.new(val){ 0 }]))
            next if index == layers.length - 1

            a = layers[index]
            b = layers[index + 1]

            @weights.push(
                Matrix.build(b, a) {
                    |row, col|
                    args[:startingWeightFunction].call(a)
                }
            )
        }
    end

    def defaults
        {
            :learningRate => 1,
            :activationFunction => SIGMOID,
            :derivativeActivationFunction => DERIVATIVE_SIGMOID,
            :startingWeightFunction => INVERSE_SQRT
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
        return avgErr
    end

    def feedForward(input)
        layer = 0
        @weightedInputs = [input]
        output = mapActivationFunction(input + @biases[layer])

        weights.each { |weightLayer|
            layer += 1
            output = weightLayer * output + @biases[layer]
            @weightedInputs.push(output)
            output = mapActivationFunction(output)
        }
        return output
    end

    # actual and expected are matrices
    def backPropogation(actual, expected)
        nextError = (actual - expected).hadamard mapDerivativeActivationFunction(@weightedInputs.pop) # output error

        i = @weights.length - 1
        j = @weights.length

        until i < 0 do
            @biases[j] = @biases[j] - (nextError * @learningRate)
            weightLayer = @weights[i]

            z = @weightedInputs.pop
            a = mapActivationFunction(z)

            error = (weightLayer.t * nextError).hadamard mapDerivativeActivationFunction(z) # hidden error
            deltaW = nextError * a.t * @learningRate

            @weights[i] = weightLayer - deltaW

            nextError = error
            i -= 1
            j -= 1
        end
        @biases[j] = @biases[j] - (nextError * @learningRate)
    end

    def loadFromString(string)
        values = Marshal.load(string)
        @weights = values[:weights]
        @biases = values[:biases]
    end

    def saveToString
        Marshal.dump({weights: @weights, biases: @biases})
    end

    def to_s
        saveToString
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

class Matrix
    # elementwise multiplication
    def hadamard(m)
      case m
      when Numeric
        Matrix.Raise ErrOperationNotDefined, "element_multiplication", self.class, m.class
      when Vector
        m = self.class.column_vector(m)
      when Matrix
      else
        return apply_through_coercion(m, __method__)
      end

      Matrix.Raise ErrDimensionMismatch unless row_count == m.row_count && column_count == m.column_count

      rows = Array.new(row_count) do |i|
        Array.new(column_count) do|j|
          self[i, j] * m[i, j]
        end
      end
      new_matrix rows, column_count
    end
end
