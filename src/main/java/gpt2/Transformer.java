package gpt2;

import java.util.*;

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
        int nextToken = selectNextToken(embedding);
        result.add(nextToken);

        // Now we have to use the transformer again an again, getting new and new tokens, for input passing the previous output
        for (pos++; pos < config.maxLength + inputTokens.size(); pos++)
        {
            // Add the previously generated new token as input
            embedding = processToken(pos, nextToken);

            // The output will be the next new token
            nextToken = selectNextToken(embedding);
            result.add(nextToken);

            // Exit if the END_OF_TEXT token was chosen or the context size is reached
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
     * Determine the next token, based on the output of the transformer
     *
     * @param output - The output of the transformer
     * @return the token id of the new token
     */
    private int selectNextToken(float[] output)
    {
        // During the training the expected output was the word token embedding of the next token.
        // (The parameters were tuned based on the difference to the actual and expected output.)
        // That's why we hope, the result of a trained system will be a word token embedding. (The embedding of the "best" next token.)

        // In reality the output won't be matching perfectly to any of the existing embeddings,
        // but it can be similar to some of these. We have to determine how similar the output to every known token.

        // This similarity-check can be implemented by a simple dot product calculation (multiplying each position and sum up),
        // because a similar vector will have at least the same sign at all values,
        // so the multiplication will be positive mostly (negative times negative and positive times positive as well positive)
        // while a dot product with a less similar vector will contain negative elements as well.
        // The consequence is that, the dot product with a more similar vector will be a higher number, comparing to a less similar one.

        // Here we calculate the dot product with all token embeddings in a single vector - matrix multiplication
        // The output is a vector (matrix with a single row), so the dimensions are: 1 * embeddingSize
        // The transposed token embedding matrix has a dimension of embeddingSize * 50257
        // That's why the result of the matrix multiplication will have a dimension of 1 * 50257, so simply 50257 numbers
        // This number (logit) will be higher, if the particular token is more similar to the output

        // We can transform the logit to probability, using the softmax function7
        // logit = ln(probability / 1 - probability)

        float[] logits = Util.multiplyVectorByTransposedMatrix(output, params.tokenEmbeddings);

        // It would be possible to implement here the temperature and topP filter as well:
        // temperature: divide the logits by the temperature (value between 0 and 1)
        // topP filter: on an ordered list of probabilities filter out the remaining after reaching a certain sum percentage

        // But now only the topK filtering will be implemented, so we will select a token of the best k possibilities

        // Converting the list of logits, ordered by token id into an ordered set (using a custom comparator on the logit),
        // wrapping the token id and logit into a single object (TokenWithLogitComparator):

        TreeSet<TokenLogit> orderedTokenLogits = new TreeSet<>(new TokenLogitComparator());
        for (int i = 0; i < logits.length; i++)
        {
            orderedTokenLogits.add(new TokenLogit(i, logits[i]));
        }

        // Retain only the top k elements (filtering out the rest)
        float[] filteredLogits = new float[config.topK];
        TokenLogit[] filteredTokenLogits = new TokenLogit[config.topK];

        int i = 0;
        for (TokenLogit indexedLogit : orderedTokenLogits)
        {
            filteredLogits[i] = indexedLogit.logit;
            filteredTokenLogits[i] = indexedLogit;

            i++;
            if (i == config.topK) break;
        }

        // Convert the logits into probabilities, using softmax
        float[] probabilities = Util.softmax(filteredLogits);

        // Pick one token randomly, using a weighted random selection.
        int index = Util.weightedRandomPick(probabilities);

        // Lookup the token id
        int selectedTokenId = filteredTokenLogits[index].tokenId;

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

    private static class TokenLogit
    {
        public int tokenId;
        public float logit;

        public TokenLogit(int tokenId, float logit)
        {
            this.tokenId = tokenId;
            this.logit = logit;
        }
    }

    private static class TokenLogitComparator implements Comparator<TokenLogit>
    {
        public int compare(TokenLogit a, TokenLogit b)
        {
            return a.logit > b.logit ? -1 : 1;
        }
    }
}
