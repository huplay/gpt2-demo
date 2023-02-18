package ai.demo.gpt2;

import ai.demo.gpt2.util.Util;

import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.*;
import static java.lang.Math.pow;

public class TransformerDecoder
{
    private final int embeddingSize;
    private final int headCount;
    private final int attentionDividend;

    private final Parameters.DecoderParameters params;
    private final Util util;

    private final float epsilon;

    private final List<float[][]> storedKeys = new ArrayList<>();
    private final List<float[][]> storedValues = new ArrayList<>();

    public TransformerDecoder(Config config, Parameters.DecoderParameters params, float epsilon)
    {
        this.embeddingSize = config.modelType.embeddingSize;
        this.headCount = config.modelType.headCount;

        // The vector size is always 64, so this is always 8, it is possible to convert to int.
        this.attentionDividend = (int) sqrt(embeddingSize / headCount);

        this.params = params;
        this.util = config.utilType.util;
        this.epsilon = epsilon;
    }

    /**
     * Decoder logic
     *
     * @param input - input embedding
     * @return output embedding
     */
    public float[] calculate(float[] input)
    {
        // Attention block
        float[] result = attentionBlock(input);

        // Feed forward block
        return feedForwardBlock(result);
    }

    private float[] attentionBlock(float[] input)
    {
        // Normalization
        float[] result = normalize(input, params.norm1Weights, params.norm1Biases);

        // Attention layer
        result = attention(result);

        // Residual connection
        return util.addVectors(result, input);
    }

    private float[] normalize(float[] input, float[] weights, float[] biases)
    {
        // Standard normalization
        float[] result = util.normalize(input, epsilon);

        // Applying the trained weights and biases
        for (int i = 0; i < input.length; i++)
        {
            result[i] = result[i] * weights[i] + biases[i];
        }

        return result;
    }

    private float[] feedForwardBlock(float[] input)
    {
        // Normalization
        float[] result = normalize(input, params.norm2Weights, params.norm2Biases);

        // Feed forward layers
        result = feedForward(result);

        // Residual connection
        return util.addVectors(result, input);
    }

    private float[] attention(float[] embedding)
    {
        // Calculate the query, key and value vectors for the actual token:

        // params.attentionWeighs contains 3 matrices (WQ, WK and WV), concatenated into a single matrix
        // params.attentionBiases similarly contains 3 vectors
        // The query, key and value matrices for the actual token is created using the actual embedding value
        // and the WQ, WK an WV matrices (+biases):
        float[] weighted = util.multiplyVectorByMatrix(embedding, params.attentionWeighs); // [3 x embeddingSize]
        float[] biased = util.addVectors(weighted, params.attentionBiases); // [3 x embeddingSize]

        // Split the result matrix into 3 parts for the query, key and values:
        float[][] segments = util.splitVector(biased, 3); // [3][embeddingSize]

        // Split the query, key and value vectors into pieces for all heads
        float[][] queries = util.splitVector(segments[0], headCount); // [headCount][embeddingSize / headCount]
        float[][] keys = util.splitVector(segments[1], headCount); // [headCount][embeddingSize / headCount]
        float[][] values = util.splitVector(segments[2], headCount); // [headCount][embeddingSize / headCount]

        // Store the keys and values (these will be available while the following tokens will be processed)
        storedKeys.add(keys);
        storedValues.add(values);

        float[][] sums = new float[headCount][embeddingSize / headCount];

        // Scoring the previous tokens (including the actual), separately for all heads
        // Again: we have to score not only the previous, but the actual token as well
        // That is the reason of that we already added the actual key/value to the stored keys/values
        for (int head = 0; head < headCount; head++)
        {
            // Calculate the scores
            float[] scores = new float[storedKeys.size()];
            for (int pos = 0; pos < storedKeys.size(); pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                scores[pos] = util.dotProduct(queries[head], storedKeys.get(pos)[head]) / attentionDividend;
            }

            // Softmax
            scores = util.softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < storedKeys.size(); pos++)
            {
                float[] sum = util.multiplyVectorByScalar(storedValues.get(pos)[head], scores[pos]);
                sums[head] = util.addVectors(sums[head], sum);
            }
        }

        // Concatenate the results for all heads
        float[] flatSums = util.flattenMatrix(sums);

        // Apply the attention projection weights and biases
        float[] result = util.multiplyVectorByMatrix(flatSums, params.attentionProjectionWeights);

        return util.addVectors(result, params.attentionProjectionBiases);
    }

    private float[] feedForward(float[] embedding)
    {
        // We have a simple feed forward neural network, which has only two layers:
        // - the first layer has 4 x <embeddingSize> neurons (using a gelu activation function)
        // - the second layer has <embeddingSize> neurons (without activation function, simply resulting the weighted + biased input)

        // The calculation of a feed forward neutron layer is simply a multiplication of the input vector by the weight matrix,
        // and a vector to vector addition using the biases

        // First layer
        float[] output = util.multiplyVectorByMatrix(embedding, params.layer1Weights);
        output = util.addVectors(output, params.layer1Biases);

        // Using the gelu activation function, calculating the output of the first layer
        for (int neuron = 0; neuron < 4 * embeddingSize; neuron++)
        {
            output[neuron] = gelu(output[neuron]);
        }

        // Second layer (no activation function call)
        output = util.multiplyVectorByMatrix(output, params.layer2Weights);
        return util.addVectors(output, params.layer2Biases);
    }

    // Gaussian Error Linear Unit (GELU) cumulative distribution activation function (approximate implementation)
    private static float gelu(float value)
    {
        return (float) (0.5 * value * (1 + tanh(sqrt(2 / PI) * (value + 0.044715 * pow(value, 3)))));
    }

    /**
     * Clear stored values to start a new session
     */
    public void clear()
    {
        storedKeys.clear();
        storedValues.clear();
    }
}
