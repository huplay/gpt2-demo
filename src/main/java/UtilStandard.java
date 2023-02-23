import static java.lang.Math.exp;
import static java.lang.Math.sqrt;

public class UtilStandard implements Util
{
    @Override
    public float[] addVectors(float[] vector1, float[] vector2)
    {
        float[] ret = new float[vector1.length];

        for (int i = 0; i < vector1.length; i++)
        {
            ret[i] = vector1[i] + vector2[i];
        }

        return ret;
    }

    @Override
    public float dotProduct(float[] vector1, float[] vector2)
    {
        float sum = 0;

        for (int i = 0; i < vector1.length; i++)
        {
            sum = sum + vector1[i] * vector2[i];
        }

        return sum;
    }

    @Override
    public float[] multiplyVectorByScalar(float[] vector, float scalar)
    {
        float[] ret = new float[vector.length];

        for (int i = 0; i < vector.length; i++)
        {
            ret[i] = vector[i] * scalar;
        }

        return ret;
    }

    @Override
    public float[] multiplyVectorByMatrix(float[] vector, float[][] matrix)
    {
        float[] ret = new float[matrix[0].length];

        for (int i = 0; i < matrix[0].length; i++)
        {
            for (int j = 0; j < vector.length; j++)
            {
                float sum = 0;

                for (int x = 0; x < vector.length; x++)
                {
                    sum = sum + vector[x] * matrix[x][i];
                }

                ret[i] = sum;
            }
        }

        return ret;
    }

    @Override
    public float[] multiplyVectorByTransposedMatrix(float[] vector, float[][] matrix)
    {
        float[] ret = new float[matrix.length];

        for (int i = 0; i < matrix.length; i++)
        {
            for (int j = 0; j < vector.length; j++)
            {
                float sum = 0;

                for (int x = 0; x < vector.length; x++)
                {
                    sum = sum + vector[x] * matrix[i][x];
                }

                ret[i] = sum;
            }
        }

        return ret;
    }

    @Override
    public float[][] splitVector(float[] vector, int count)
    {
        int size = vector.length / count;
        float[][] ret = new float[count][size];

        int segment = 0;
        int col = 0;
        for (float value : vector)
        {
            ret[segment][col] = value;

            if (col == size - 1)
            {
                col = 0;
                segment++;
            }
            else
            {
                col++;
            }
        }

        return ret;
    }

    @Override
    public float[] flattenMatrix(float[][] matrix)
    {
        float[] ret = new float[matrix.length * matrix[0].length];

        int i = 0;

        for (float[] row : matrix)
        {
            for (float value : row)
            {
                ret[i] = value;
                i++;
            }
        }

        return ret;
    }

    @Override
    public float average(float[] vector)
    {
        double sum = 0;

        for (float value : vector)
        {
            sum = sum + value;
        }

        return (float) sum / vector.length;
    }

    @Override
    public float[] softmax(float[] vector)
    {
        double total = 0;

        for (float value : vector)
        {
            total = total + exp(value);
        }

        float[] ret = new float[vector.length];

        for (int i = 0; i < vector.length; i++)
        {
            ret[i] = (float) (exp(vector[i]) / total);
        }

        return ret;
    }

    /**
     * Standard normalization:
     *    (value - avg) * sqrt( (value - avg)^2 + epsilon )
     *
     * @param vector - input vector
     * @param epsilon - epsilon in the formula above
     * @return the normalized vector
     */
    @Override
    public float[] normalize(float[] vector, float epsilon)
    {
        float average = average(vector);
        float averageDiff = calculateAverageDiff(vector, average, epsilon);

        float[] norm = new float[vector.length];

        for (int i = 0; i < vector.length; i++)
        {
            norm[i] = (vector[i] - average) / averageDiff;
        }

        return norm;
    }

    private float calculateAverageDiff(float[] values, float average, float epsilon)
    {
        float[] squareDiff = new float[values.length];

        for (int i = 0; i < values.length; i++)
        {
            float diff = values[i] - average;
            squareDiff[i] = diff * diff;
        }

        float averageSquareDiff = average(squareDiff);

        return (float) sqrt(averageSquareDiff + epsilon);
    }

    @Override
    public float[][] sort(float[][] matrix)
    {
        // TODO: Instead of a standard implementation I simply called the Nd4j solution
        return UtilType.ND4J.util.sort(matrix);
    }

    @Override
    public int weightedRandomPick(float[] probabilities)
    {
        float sum = 0;
        float[] cumulativeProbabilities = new float[probabilities.length];

        for (int i = 0; i < probabilities.length; i++)
        {
            sum = sum + probabilities[i] * 100;
            cumulativeProbabilities[i] = sum;
        }

        int random = (int)(Math.random() * sum);

        int index = 0;
        for (int i = 0; i < probabilities.length; i++)
        {
            if (random < cumulativeProbabilities[i]) break;

            index ++;
        }

        return index;
    }
}
