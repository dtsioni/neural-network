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
