import java.util.ArrayList;
import java.util.List;

/**
 * Decoder-only Transformer implementation, same architecture as OpenAI GPT-2
 */
public class Transformer
{
    private static final int END_OF_TEXT = 50256;
    private static final float EPSILON = 1e-5f;

    private final Config config;
    private final Parameters params;
    private final Util util;

    private final TransformerDecoder[] decoders;

    /**
     * Constructor for the Transformer (initialize values)
     */
    public Transformer(Config config, Parameters params)
    {
        this.config = config;
        this.params = params;
        this.util = config.utilType.util;

        // Create the decoder stack
        this.decoders = new TransformerDecoder[config.modelType.decoderCount];
        for (int i = 0; i < config.modelType.decoderCount; i++)
        {
            this.decoders[i] = new TransformerDecoder(config, params.decoderParameters[i], EPSILON);
        }
    }

    /**
     * Transformer token processing logic
     *
     * This method implements the logic how the input tokens and the new and new generated tokens are passed to the transformer
     *
     * @param inputTokens List of tokens, representing the provided input text.
     * @return - list of generated tokens.
     */
    public List<Integer> processTokens(List<Integer> inputTokens)
    {
        // Collector of the generated new tokens
        List<Integer> result = new ArrayList<>();

        // Counter of the position of the tokens within the text
        int pos = 0;

        if (inputTokens.size() == 0)
        {
            // If the input is empty, use the END_OF_TEXT token as input
            inputTokens.add(END_OF_TEXT);
        }
        else
        {
            // Iterating over on the input tokens (excluding the last one) and processing these by the transformer
            for (; pos < inputTokens.size() - 1; pos++)
            {
                // We are not interested in the result of the process (no return value),
                // but the inner state will be stored within the decoders (generated key and value vectors)
                processToken(pos, inputTokens.get(pos));
            }
        }

        // Processing the last input token. The output will be the first new token
        float[] embedding = processToken(pos, inputTokens.get(pos));
        int nextToken = determineOutput(embedding);
        result.add(nextToken);

        // Now we have to use the transformer again an again, getting new and new tokens, for input passing the previous output
        for (pos++; pos < config.maxLength + inputTokens.size(); pos++)
        {
            // Add the previously generated new token as input
            embedding = processToken(pos, nextToken);

            // The output will be the next new token
            nextToken = determineOutput(embedding);
            result.add(nextToken);

            // Exit if the end-of-text token was chosen or the context size is reached
            if (nextToken == END_OF_TEXT || (inputTokens.size() + result.size() >= config.modelType.contextSize)) break;
        }

        return result;
    }

    /**
     * Transformer logic for a single token processing
     *
     * @param pos - Position of the token within the provided text
     * @param token - The token id of the actual token
     * @return the embedding (representation) of the token after it is processed by the transformer
     * (Meantime the inner state will be stored, so subsequent transformer executions will use the key and value matrices of the previous tokens)
     */
    private float[] processToken(int pos, int token)
    {
        // Word token embedding
        float[] embedding = params.tokenEmbeddings[token];

        // Position embedding
        embedding = util.addVectors(embedding, params.positionEmbeddings[pos]);

        // Decoder stack
        for (TransformerDecoder decoder : decoders)
        {
            embedding = decoder.calculate(embedding);
        }

        // Final normalization
        embedding = util.normalize(embedding, EPSILON);
        for (int i = 0; i < embedding.length; i++)
        {
            embedding[i] = embedding[i] * params.normFinalWeights[i] + params.normFinalBiases[i];
        }

        return embedding;
    }

    /**
     * Determine the next token, based on the output of the transformer
     *
     * @param embedding - The output of the transformer
     * @return the token id of the new token
     */
    private int determineOutput(float[] embedding)
    {
        // Transforming the embedding into token probabilities*, using a simple matrix multiplication
        // The embedding is a vector (matrix with a single row), so the dimensions are: 1 x embeddingSize
        // The transposed token embedding matrix has a dimension of embeddingSize x 50257
        // That's why the result of the matrix multiplication will have a dimension of 1 x 50257, so simply 50257 numbers
        // This will be treated as a list of token probabilities, ordered by the token id

        // * This isn't a real probability, but similar, called logit. logit = ln(probability / 1 - probability)
        //   This value can be negative as well, but lower value means lower probability,
        //   so we can use this value filtering out the small probabilities
        float[] logits = util.multiplyVectorByTransposedMatrix(embedding, params.tokenEmbeddings);

        // topK filtering

        // It would be possible to implement here the temperature and topP filter as well:
        // temperature: divide the logits by the temperature (value between 0 and 1)
        // topP filter: on an ordered list of probabilities filter out the remaining after reaching a certain sum percentage

        // Converting the list of logits, ordered by token id into a matrix with two columns:
        // The first column is the logits, the second column is the token id (index)
        float[][] indexedLogits = new float[logits.length][2];
        for (int i = 0; i < logits.length; i++)
        {
            indexedLogits[i][0] = logits[i];
            indexedLogits[i][1] = i;
        }

        // Now we can sort the rows in this matrix, and the token id information will remain available
        float[][] sortedLogits = util.sort(indexedLogits);

        // Retain the top k elements (filtering out the rest)
        // We can omit the token id because we can find it in the sortedLogits array
        float[] filteredLogits = new float[config.topK];
        for (int i = 0; i < config.topK; i++)
        {
            filteredLogits[i] = sortedLogits[i][0];
        }

        // Convert the logits into probabilities, using softmax
        float[] probabilities = util.softmax(filteredLogits);

        // Pick one token randomly, using a weighted random selection.
        int index = util.weightedRandomPick(probabilities);

        // Lookup the token id
        int selectedTokenId = (int)sortedLogits[index][1];

        // Print the generated tokens one by one.
        // This isn't a perfect solution, because some words or letters represented by multiple tokens.
        // But the system is slow, it's better to see the progress than waiting till the end.
        Application.OUT.print(config.tokenizer.decode(List.of(selectedTokenId)));

        return selectedTokenId;
    }

    /**
     * Clear stored values in all decoders to start a new session
     */
    public void clear()
    {
        for (TransformerDecoder decoder : decoders)
        {
            decoder.clear();
        }
    }
}
