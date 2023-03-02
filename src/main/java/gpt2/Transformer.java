package gpt2;

import java.util.*;
import static gpt2.Util.IndexedValue;

/**
 * Decoder-only Transformer implementation, same architecture as OpenAI GPT-2
 */
public class Transformer
{
    private static final int END_OF_TEXT = 50256;
    private static final float EPSILON = 1e-5f;

    private final Config config;
    private final Parameters params;

    private final TransformerDecoder[] decoders;

    /**
     * Constructor for the Transformer (initialize values)
     */
    public Transformer(Config config, Parameters params)
    {
        this.config = config;
        this.params = params;

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
        int intputSize = inputTokens.size();

        if (intputSize == 0)
        {
            // If the input is empty, use the END_OF_TEXT token as input
            inputTokens.add(END_OF_TEXT);
            intputSize = 1;
        }
        else
        {
            // Iterating over on the input tokens (excluding the last one) and processing these by the transformer
            // We are not interested in the output of the transformer, but the inner state will be stored
            for (int pos = 0; pos < intputSize - 1; pos++)
            {
                processToken(pos, inputTokens.get(pos));
            }
        }

        // Collector of the generated new tokens
        List<Integer> result = new ArrayList<>();

        int nextToken = inputTokens.get(intputSize - 1);

        // Use the transformer again an again to generate new tokens
        for (int pos = intputSize - 1; pos < config.maxLength + intputSize; pos++)
        {
            // Add the last input token or the previously generated new token as input
            float[] output = processToken(pos, nextToken);

            // The output will be the next new token
            nextToken = selectNextToken(output);
            result.add(nextToken);

            // Exit if the END_OF_TEXT token was chosen or the context size is reached
            if (nextToken == END_OF_TEXT || (intputSize + result.size() >= config.modelType.contextSize)) break;
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
        float[] hiddenState = params.tokenEmbeddings[token];

        // Position embedding
        hiddenState = Util.addVectors(hiddenState, params.positionEmbeddings[pos]);

        // Decoder stack
        for (TransformerDecoder decoder : decoders)
        {
            hiddenState = decoder.execute(hiddenState);
        }

        // Final normalization
        hiddenState = Util.normalize(hiddenState, EPSILON);
        for (int i = 0; i < hiddenState.length; i++)
        {
            hiddenState[i] = hiddenState[i] * params.normFinalWeights[i] + params.normFinalBiases[i];
        }

        return hiddenState;
    }

    /**
     * Select the next token, based on the output of the transformer
     *
     * @param output - The output of the transformer
     * @return the token id of the selected new token
     */
    private int selectNextToken(float[] output)
    {
        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        float[] logits = Util.multiplyVectorByTransposedMatrix(output, params.tokenEmbeddings);

        // BTW: It would be possible to implement the temperature and topP filter as well

        // Sort (higher to lower) the result of the dot products, retaining the order (index) of the related token
        List<IndexedValue> orderedLogits = Util.reverseAndFilter(logits, config.topK);

        // Convert the logits to probabilities
        float[] probabilities = Util.softmax(orderedLogits);

        // Pick one token randomly, using a weighted random selection.
        int index = Util.weightedRandomPick(probabilities);

        // Lookup the token id
        int selectedTokenId = orderedLogits.get(index).index;

        // Print the generated token - It isn't perfect, because some words or letters represented by multiple tokens.
        // But it's better to see the progress than waiting till the end.
        Application.OUT.print(config.tokenizer.decode(Collections.singletonList(selectedTokenId)));

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
