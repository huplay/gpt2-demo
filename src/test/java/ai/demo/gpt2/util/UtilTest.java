package ai.demo.gpt2.util;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class UtilTest
{
    public static void addVectorsTest(Util util)
    {
        float[] a = {1, 2, 3, 4};
        float[] b = {4, 5, 6, 7};
        float[] expectedResult = {5, 7, 9, 11};

        assertArrayEquals(expectedResult, util.addVectors(a, b), 0);
    }

    public static void multiplyVectorByScalarTest(Util util)
    {
        float[] a = {5, 6, 7, 8};
        float[] expectedResult = {15, 18, 21, 24};

        assertArrayEquals(expectedResult, util.multiplyVectorByScalar(a, 3), 0);
    }

    public static void dotProductTest(Util util)
    {
        float[] a = {5, 6, 7, 8};
        float[] b = {4, 5, 6, 7};

        assertEquals(5*4 + 6*5 + 7*6 + 8*7, util.dotProduct(a, b), 0);
    }

    public static void multiplyVectorByMatrixTest(Util util)
    {
        float[] a = {5, 6, 7, 8};
        float[][] b = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}};

        float[] expectedResult = {5 + 6*4 + 7*7 + 8*10, 5*2 + 6*5 + 7*8 + 8*11, 5*3 + 6*6 + 7*9 + 8*12};

        assertArrayEquals(expectedResult, util.multiplyVectorByMatrix(a, b), 0);
    }

    public static void multiplyVectorByTransposedMatrixTest(Util util)
    {
        float[] a = {5, 6, 7, 8};
        float[][] b = {
                {1, 4, 7, 10},
                {2, 5, 8, 11},
                {3, 6, 9, 12}};

        float[] expectedResult = {5 + 6*4 + 7*7 + 8*10, 5*2 + 6*5 + 7*8 + 8*11, 5*3 + 6*6 + 7*9 + 8*12};

        assertArrayEquals(expectedResult, util.multiplyVectorByTransposedMatrix(a, b), 0);
    }

    public static void splitVectorTest(Util util)
    {
        float[] matrix = {1, 2, 3, 4, 5, 6};
        float[][] expectedResult = {{1, 2}, {3, 4}, {5, 6}};

        assertArrayEquals(expectedResult, util.splitVector(matrix, 3));
    }

    public static void flattenMatrixTest(Util util)
    {
        float[][] matrix = {{1, 2}, {3, 4}, {5, 6}};
        float[] expectedResult = {1, 2, 3, 4, 5, 6};

        assertArrayEquals(expectedResult, util.flattenMatrix(matrix), 0);
    }

    public static void averageTest(Util util)
    {
        float[] matrix = {1, 2, 3, 4, 5, 6};

        assertEquals(3.5f, util.average(matrix), 0);
    }

    public static void softmaxTest(Util util)
    {
        float[] values = new float[] {1, 2, 3, 4, 1, 2, 3};

        float[] expected = new float[] {0.023640543f, 0.06426166f, 0.1746813f, 0.474833f, 0.023640543f, 0.06426166f, 0.1746813f};

        assertArrayEquals(expected, util.softmax(values), 0);
    }
}
