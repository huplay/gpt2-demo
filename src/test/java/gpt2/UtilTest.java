package gpt2;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class UtilTest
{
    @Test
    public void addVectorsTest()
    {
        float[] a = {1, 2, 3, 4};
        float[] b = {4, 5, 6, 7};
        float[] expectedResult = {5, 7, 9, 11};

        assertArrayEquals(expectedResult, Util.addVectors(a, b), 0);
    }

    @Test
    public void multiplyVectorByScalarTest()
    {
        float[] a = {5, 6, 7, 8};
        float[] expectedResult = {15, 18, 21, 24};

        assertArrayEquals(expectedResult, Util.multiplyVectorByScalar(a, 3), 0);
    }

    @Test
    public void dotProductTest()
    {
        float[] a = {5, 6, 7, 8};
        float[] b = {4, 5, 6, 7};

        assertEquals(5*4 + 6*5 + 7*6 + 8*7, Util.dotProduct(a, b), 0);
    }

    @Test
    public void multiplyVectorByMatrixTest()
    {
        float[] a = {5, 6, 7, 8};
        float[][] b = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}};

        float[] expectedResult = {5 + 6*4 + 7*7 + 8*10, 5*2 + 6*5 + 7*8 + 8*11, 5*3 + 6*6 + 7*9 + 8*12};

        assertArrayEquals(expectedResult, Util.multiplyVectorByMatrix(a, b), 0);
    }

    @Test
    public void multiplyVectorByTransposedMatrixTest()
    {
        float[] a = {5, 6, 7, 8};
        float[][] b = {
                {1, 4, 7, 10},
                {2, 5, 8, 11},
                {3, 6, 9, 12}};

        float[] expectedResult = {5 + 6*4 + 7*7 + 8*10, 5*2 + 6*5 + 7*8 + 8*11, 5*3 + 6*6 + 7*9 + 8*12};

        assertArrayEquals(expectedResult, Util.multiplyVectorByTransposedMatrix(a, b), 0);
    }

    @Test
    public void splitVectorTest()
    {
        float[] matrix = {1, 2, 3, 4, 5, 6};
        float[][] expectedResult = {{1, 2}, {3, 4}, {5, 6}};

        assertArrayEquals(expectedResult, Util.splitVector(matrix, 3));
    }

    @Test
    public void flattenMatrixTest()
    {
        float[][] matrix = {{1, 2}, {3, 4}, {5, 6}};
        float[] expectedResult = {1, 2, 3, 4, 5, 6};

        assertArrayEquals(expectedResult, Util.flattenMatrix(matrix), 0);
    }

    @Test
    public void averageTest()
    {
        float[] matrix = {1, 2, 3, 4, 5, 6};

        assertEquals(3.5f, Util.average(matrix), 0);
    }

    @Test
    public void softmaxTest()
    {
        float[] values = new float[] {1, 2, 3, 4, 1, 2, 3};

        float[] expected = new float[] {0.023640543f, 0.06426166f, 0.1746813f, 0.474833f, 0.023640543f, 0.06426166f, 0.1746813f};

        assertArrayEquals(expected, Util.softmax(values), 0);
    }

    @Test
    public void normalizeTest()
    {
        float[] values = new float[] {1, 2, 3, 4, 1, 2, 3};

        float[] expected = new float[] {-1.2480696f, -0.2773489f, 0.6933719f, 1.6640927f, -1.2480696f, -0.2773489f, 0.6933719f};

        assertArrayEquals(expected, Util.normalize(values, 1e-5f), 0);
    }
}
